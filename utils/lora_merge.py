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
        r"^(?P<name>.+)\.(?P<type>(?:lora|lycoris)_(?:down|up|mid))\.weight$"
    )

    for lora_sd, ratio in zip(loras, ratios):
        keys = list(lora_sd.keys())
        for key in keys:
            m = pattern.match(key)
            if not m:
                print(f"[merge_lora] skip unmatched key: {key}")
                continue
            name = m.group("name")
            typ = m.group("type")

            base = name
            down_key = None
            up_key = None
            mid_key = None

            if typ.endswith("down"):
                down_key = key
                up_key = name + "." + typ.replace("down", "up") + ".weight"
                mid_key = name + "." + typ.replace("down", "mid") + ".weight"
            else:
                continue

            if up_key not in lora_sd:
                print(f"[merge_lora] missing up weight for {down_key}")
                continue

            down = lora_sd[down_key].to(dtype)
            up = lora_sd[up_key].to(dtype)
            mid = lora_sd.get(mid_key)

            dim = down.size(0)
            alpha = lora_sd.get(name + ".alpha", dim)
            scale = alpha / dim

            t_map, t_sd, mod_name = route_name(base)
            mod_name = mod_name.replace(".", "_")

            if t_map is None or mod_name not in t_map:
                print(f"[merge_lora] no module found for {mod_name} ({key})")
                continue

            base_key = t_map[mod_name]
            weight = t_sd[base_key].to(dtype)

            if mid is not None:
                wa = up.view(up.size(0), -1).transpose(0, 1)
                wb = down.view(down.size(0), -1)
                update = torch.einsum("ij...,ip,jr->pr...", mid.to(dtype), wa, wb)
                update = update.view_as(weight) * scale * ratio
            elif len(weight.shape) == 2:
                if len(up.shape) == 4:
                    up = up.squeeze(3).squeeze(2)
                    down = down.squeeze(3).squeeze(2)
                update = torch.matmul(up, down) * scale * ratio
            elif down.dim() == 4 and tuple(down.shape[2:]) == (1, 1):
                update = (
                    torch.matmul(up.squeeze(3).squeeze(2), down.squeeze(3).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                ) * scale * ratio
            else:
                conv = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
                update = conv * scale * ratio

            t_sd[base_key] = (weight + update).to(dtype)

    return unet, clip_l, clip_g
