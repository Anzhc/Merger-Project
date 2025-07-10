import torch
from typing import Dict, List, Tuple, Optional

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

    for lora_sd, ratio in zip(loras, ratios):
        for key in list(lora_sd.keys()):
            if not key.endswith("lora_down.weight"):
                continue
            base = key[:-len("lora_down.weight")]
            up_key = base + "lora_up.weight"
            alpha_key = base + "alpha"

            if up_key not in lora_sd:
                continue

            down = lora_sd[key].to(dtype)
            up = lora_sd[up_key].to(dtype)
            dim = down.size(0)
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim

            target_map = None
            target_sd = None
            name = base

            if name.startswith(LORA_PREFIX_UNET + "_"):
                if not merge_unet:
                    continue
                name = name[len(LORA_PREFIX_UNET) + 1 :]
                target_map = unet_map
                target_sd = unet
            elif name.startswith(LORA_PREFIX_TEXT_ENCODER1 + "_"):
                if not merge_clip_l:
                    continue
                name = name[len(LORA_PREFIX_TEXT_ENCODER1) + 1 :]
                target_map = clip_l_map
                target_sd = clip_l
            elif name.startswith(LORA_PREFIX_TEXT_ENCODER2 + "_"):
                if not merge_clip_g:
                    continue
                name = name[len(LORA_PREFIX_TEXT_ENCODER2) + 1 :]
                target_map = clip_g_map
                target_sd = clip_g
            elif name.startswith(LORA_PREFIX_TEXT_ENCODER + "_"):
                if not merge_clip_l:
                    continue
                name = name[len(LORA_PREFIX_TEXT_ENCODER) + 1 :]
                target_map = clip_l_map
                target_sd = clip_l
            else:
                continue

            if target_map is None or name not in target_map:
                continue
            base_key = target_map[name]
            weight = target_sd[base_key].to(dtype)

            if len(weight.shape) == 2:
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
            target_sd[base_key] = (weight + update).to(dtype)

    return unet, clip_l, clip_g
