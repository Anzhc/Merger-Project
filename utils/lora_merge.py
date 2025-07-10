import torch
from typing import Dict, List, Tuple, Optional
import re

# Prefixes used by kohya-style LoRA weights
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"


def _build_key_map(state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Map normalized parameter names to actual keys."""
    mapping = {}
    for key in state_dict.keys():
        base = key
        if key.endswith(".weight"):
            base = key[:-7]
        elif key.endswith(".bias"):
            base = key[:-5]
        mapping[base.replace(".", "_")] = key
    return mapping


def merge_lora_models(
    unet: Optional[Dict[str, torch.Tensor]],
    clip_l: Optional[Dict[str, torch.Tensor]],
    clip_g: Optional[Dict[str, torch.Tensor]],
    loras: List[Dict[str, torch.Tensor]],
    ratios: List[float],
    merge_unet: bool = True,
    merge_clip_l: bool = True,
    merge_clip_g: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    """Merge ``loras`` into the given state dicts in-place and return them."""

    unet_map = _build_key_map(unet) if unet is not None else {}
    clip_l_map = _build_key_map(clip_l) if clip_l is not None else {}
    clip_g_map = _build_key_map(clip_g) if clip_g is not None else {}

    def route_name(name: str):
        """Return target map and module name for a LoRA prefix."""
        target_map, target_sd = unet_map, unet

        if name.startswith((LORA_PREFIX_TEXT_ENCODER1 + "_", "lycoris_te1_")):
            target_map, target_sd = clip_l_map, clip_l
            name = name.split("_", 1)[1]
        elif name.startswith((LORA_PREFIX_TEXT_ENCODER2 + "_", "lycoris_te2_")):
            target_map, target_sd = clip_g_map, clip_g
            name = name.split("_", 1)[1]
        elif name.startswith((LORA_PREFIX_TEXT_ENCODER + "_", "lycoris_te_")):
            target_map, target_sd = clip_l_map, clip_l
            name = name.split("_", 1)[1]
        elif name.startswith((LORA_PREFIX_UNET + "_", "lycoris_unet_")):
            target_map, target_sd = unet_map, unet
            name = name.split("_", 1)[1]
        elif name.startswith(("lora_", "lycoris_")):
            target_map, target_sd = unet_map, unet
            name = name.split("_", 1)[1]

        if name.startswith(("up_", "down_", "mid_")):
            name = name.split("_", 1)[1]

        return target_map, target_sd, name

    pattern = re.compile(
        r"^(?P<name>.+)\.(?P<type>(?:lora|lycoris)_(?:down|up|mid)|[ab][12])\.weight$"
    )

    def matmul_update(up, down, weight):
        if len(weight.shape) == 2:
            if len(up.shape) == 4:
                up = up.squeeze(3).squeeze(2)
                down = down.squeeze(3).squeeze(2)
            return torch.matmul(up, down)
        if down.dim() == 4 and tuple(down.shape[2:]) == (1, 1):
            return (
                torch.matmul(up.squeeze(3).squeeze(2), down.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
            )
        conv = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        return conv

    def loha_update(w1b, w1a, w2b, w2a):
        r = w1a.size(0)
        wa = w1b.reshape(w1b.size(0), r)
        wb = w1a.reshape(r, -1)
        wc = w2b.reshape(w2b.size(0), r)
        wd = w2a.reshape(r, -1)
        out = (wa @ wb) * (wc @ wd)
        return out.reshape(w1b.size(0), w2a.size(1), *w1a.shape[2:])

    for lora_sd, ratio in zip(loras, ratios):
        groups: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in lora_sd.items():
            if key.endswith(".alpha"):
                base = key[: -6]
                groups.setdefault(base, {})["alpha"] = tensor
                continue
            if not key.endswith(".weight"):
                continue
            m = pattern.match(key)
            if not m:
                print(f"[merge_lora] skip unmatched key: {key}")
                continue
            base = m.group("name")
            typ = m.group("type")
            part = None
            if typ.endswith("down") or typ == "a1":
                part = "down1"
            elif typ.endswith("up") or typ == "b1":
                part = "up1"
            elif typ in {"a2", "lora_mid", "lycoris_mid"}:
                part = "down2"
            elif typ == "b2":
                part = "up2"
            elif typ.endswith("mid"):
                part = "mid"
            if part is None:
                continue
            groups.setdefault(base, {})[part] = tensor

        for base, parts in groups.items():
            t_map, t_sd, mod_name = route_name(base)
            mod_name = mod_name.replace(".", "_")
            if t_map is None or mod_name not in t_map:
                print(f"[merge_lora] no module found for {mod_name} ({base})")
                continue

            weight_key = t_map[mod_name]
            weight = t_sd[weight_key].to(dtype)

            down1 = parts.get("down1")
            up1 = parts.get("up1")
            down2 = parts.get("down2")
            up2 = parts.get("up2")
            mid = parts.get("mid")

            if down1 is None or up1 is None:
                print(f"[merge_lora] missing pair for {base}")
                continue

            dim = down1.size(0)
            alpha = parts.get("alpha", dim)
            scale = alpha / dim

            if down2 is not None and up2 is not None:
                update = loha_update(up1.to(dtype), down1.to(dtype), up2.to(dtype), down2.to(dtype))
                update = update.to(dtype) * scale * ratio
                if update.shape != weight.shape:
                    update = update.view_as(weight)
            elif mid is not None:
                wa = up1.view(up1.size(0), -1).transpose(0, 1)
                wb = down1.view(down1.size(0), -1)
                update = torch.einsum("ij...,ip,jr->pr...", mid.to(dtype), wa, wb)
                update = update.view_as(weight) * scale * ratio
            else:
                update = matmul_update(up1.to(dtype), down1.to(dtype), weight) * scale * ratio

            t_sd[weight_key] = (weight + update).to(dtype)

    return unet, clip_l, clip_g
