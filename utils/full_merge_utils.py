import torch
from typing import List
import device_manager
from .peft_utils import ties, dare_linear, dare_ties, reshape_weight_task_tensors


def merge_state_dicts_with_base(base: dict, models: List[dict], weights: List[float], merge_func, **kwargs) -> dict:
    """Merge ``models`` relative to ``base`` using ``merge_func`` on deltas."""
    keys = set(base.keys())
    for d in models:
        keys.update(d.keys())

    device = torch.device(device_manager.get_device())
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    result = {}
    for k in keys:
        base_tensor = base.get(k)
        if base_tensor is None:
            continue
        deltas = []
        for d in models:
            t = d.get(k)
            if t is None:
                t = torch.zeros_like(base_tensor, device=device)
            elif not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=device)
            deltas.append(t - base_tensor)
        merged_delta = merge_func(deltas, weight_tensor, **kwargs)
        result[k] = base_tensor + merged_delta
    return result
