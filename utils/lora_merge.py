import torch
from typing import Dict, List, Tuple, Optional
import re

# Prefixes used by kohya-style LoRA weights
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"


def _build_key_map(state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Map normalized weight names to keys.

    Only parameters ending with ``.weight`` are considered to avoid
    accidentally routing LoRA tensors to a bias tensor. The keys are
    normalized by replacing ``.`` with ``_`` to match LoRA naming.
    """

    mapping = {}
    for key in state_dict.keys():
        if not key.endswith(".weight"):
            continue
        base = key[:-7]
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

        prefixes = [
            (LORA_PREFIX_TEXT_ENCODER1 + "_", clip_l_map, clip_l),
            ("lycoris_te1_", clip_l_map, clip_l),
            (LORA_PREFIX_TEXT_ENCODER2 + "_", clip_g_map, clip_g),
            ("lycoris_te2_", clip_g_map, clip_g),
            (LORA_PREFIX_TEXT_ENCODER + "_", clip_l_map, clip_l),
            ("lycoris_te_", clip_l_map, clip_l),
            (LORA_PREFIX_UNET + "_", unet_map, unet),
            ("lycoris_unet_", unet_map, unet),
            ("lora_", unet_map, unet),
            ("lycoris_", unet_map, unet),
        ]

        for pfx, mp, sd in prefixes:
            if name.startswith(pfx):
                target_map, target_sd = mp, sd
                name = name[len(pfx):]
                break

        return target_map, target_sd, name

    pattern = re.compile(
        r"^(?P<name>.+)\.(?P<type>(?:lora|lycoris)_(?:down|up|mid)|a1|a2|b1|b2|bm)\.weight$"
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

    def tucker_weight_from_conv(up, down, mid):
        up = up.reshape(up.size(0), up.size(1))
        down = down.reshape(down.size(0), down.size(1))
        return torch.einsum("m n ..., i m, n j -> i j ...", mid, up, down)

    def glora_update(weight, a1, a2, b1, b2, mid=None):
        if mid is not None:
            b_update = tucker_weight_from_conv(b1, b2, mid)
        else:
            b1_f = b1.reshape(b1.size(0), -1)
            b2_f = b2.reshape(b2.size(0), -1)
            b_update = b1_f @ b2_f
            b_update = b_update.reshape(weight.shape)

        a1_f = a1.reshape(a1.size(0), -1)
        a2_f = a2.reshape(a2.size(0), -1)
        if weight.dim() > 2:
            w_wa1 = torch.einsum("o i ..., i r -> o r ...", weight, a1_f)
            wa2 = torch.einsum("o r ..., r i -> o i ...", w_wa1, a2_f)
        else:
            wa2 = (weight @ a1_f) @ a2_f
        wa2 = wa2.reshape(weight.shape)
        return b_update + wa2

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
            if typ in {"lora_down", "lycoris_down"}:
                part = "down"
            elif typ in {"lora_up", "lycoris_up"}:
                part = "up"
            elif typ in {"lora_mid", "lycoris_mid"}:
                part = "mid"
            elif typ in {"a1", "a2", "b1", "b2", "bm"}:
                part = typ
            if part is None:
                continue
            groups.setdefault(base, {})[part] = tensor

        for base, parts in groups.items():
            t_map, t_sd, mod_name = route_name(base)
            if t_sd is unet and not merge_unet:
                continue
            if t_sd is clip_l and not merge_clip_l:
                continue
            if t_sd is clip_g and not merge_clip_g:
                continue
            mod_name = mod_name.replace(".", "_")
            if t_map is None or mod_name not in t_map:
                print(f"[merge_lora] no module found for {mod_name} ({base})")
                continue

            weight_key = t_map[mod_name]
            weight = t_sd[weight_key].to(dtype)

            down = parts.get("down")
            up = parts.get("up")
            mid = parts.get("mid")
            a1 = parts.get("a1")
            a2 = parts.get("a2")
            b1 = parts.get("b1")
            b2 = parts.get("b2")
            bm = parts.get("bm")

            if (down is None or up is None) and not (
                a1 is not None and a2 is not None and b1 is not None and b2 is not None
            ):
                print(f"[merge_lora] missing pair for {base}")
                continue

            src = down if down is not None else a1 if a1 is not None else b1
            dim = src.size(0)
            alpha = parts.get("alpha", dim)
            scale = alpha / dim

            if a1 is not None and a2 is not None and b1 is not None and b2 is not None:
                update = glora_update(weight, a1.to(dtype), a2.to(dtype), b1.to(dtype), b2.to(dtype), bm.to(dtype) if bm is not None else None)
                update = update * scale * ratio
            elif down is not None and up is not None:
                if mid is not None:
                    wa = up.view(up.size(0), -1).transpose(0, 1)
                    wb = down.view(down.size(0), -1)
                    update = torch.einsum("ij...,ip,jr->pr...", mid.to(dtype), wa, wb)
                    update = update.view_as(weight) * scale * ratio
                else:
                    update = matmul_update(up.to(dtype), down.to(dtype), weight) * scale * ratio
            else:
                print(f"[merge_lora] missing pair for {base}")
                continue

            t_sd[weight_key] = (weight + update).to(dtype)

    return unet, clip_l, clip_g
