"""
Utilities for PEFT tensor merging algorithms.

This module includes implementations taken from the HuggingFace PEFT project
(https://github.com/huggingface/peft) licensed under the Apache 2.0 License.
"""

import warnings
from typing import Literal, List

import torch
import device_manager


def reshape_weight_task_tensors(task_tensors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reshape ``weights`` to match ``task_tensors`` dimensions."""
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    return weights.view(new_shape)


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Keep top-k values of ``tensor`` based on magnitude."""
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """Randomly keep ``density`` fraction of values in ``tensor``."""
    mask = torch.bernoulli(torch.full_like(tensor, density))
    pruned = tensor * mask
    if rescale:
        pruned = pruned / density
    return pruned


def prune(tensor: torch.Tensor, density: float, method: Literal["magnitude", "random"], rescale: bool = False) -> torch.Tensor:
    """Apply pruning to ``tensor`` according to ``method``."""
    if density >= 1:
        warnings.warn(f"The density {density} is greater than or equal to 1, no pruning will be performed.")
        return tensor
    if density < 0:
        raise ValueError(f"Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    if method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(tensor: torch.Tensor, method: Literal["total", "frequency"] = "total") -> torch.Tensor:
    """Return mask of majority sign across stacked ``tensor``."""
    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    """Merge ``task_tensors`` using disjoint merge."""
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def task_arithmetic(task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    return (task_tensors * weights).sum(dim=0)


def magnitude_prune(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    pruned = [prune(t, density, method="magnitude") for t in task_tensors]
    pruned = torch.stack(pruned, dim=0)
    weights = reshape_weight_task_tensors(pruned, weights)
    return (pruned * weights).sum(dim=0)


def ties(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float, majority_sign_method: Literal["total", "frequency"] = "total") -> torch.Tensor:
    pruned = [prune(t, density, method="magnitude") for t in task_tensors]
    pruned = torch.stack(pruned, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(pruned, method=majority_sign_method)
    weights = reshape_weight_task_tensors(pruned, weights)
    weighted = pruned * weights
    return disjoint_merge(weighted, majority_sign_mask)


def dare_linear(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    pruned = [prune(t, density, method="random", rescale=True) for t in task_tensors]
    pruned = torch.stack(pruned, dim=0)
    weights = reshape_weight_task_tensors(pruned, weights)
    return (pruned * weights).sum(dim=0)


def dare_ties(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float, majority_sign_method: Literal["total", "frequency"] = "total") -> torch.Tensor:
    pruned = [prune(t, density, method="random", rescale=True) for t in task_tensors]
    pruned = torch.stack(pruned, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(pruned, method=majority_sign_method)
    weights = reshape_weight_task_tensors(pruned, weights)
    weighted = pruned * weights
    return disjoint_merge(weighted, majority_sign_mask)


def parse_weights(weights_str: str, count: int) -> List[float]:
    """Parse a comma separated list of weights."""
    if weights_str:
        try:
            weights = [float(x.strip()) for x in weights_str.split(',') if x.strip()]
        except ValueError as exc:
            raise ValueError("Weights must be numbers separated by commas") from exc
    else:
        weights = []
    if len(weights) < count:
        weights.extend([1.0] * (count - len(weights)))
    return weights[:count]


def merge_state_dicts(models: List[dict], weights: List[float], merge_func, **kwargs) -> dict:
    """Merge multiple state dicts using ``merge_func`` on each tensor key."""
    keys = set()
    for d in models:
        keys.update(d.keys())
    result = {}
    device = torch.device(device_manager.get_device())
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    for k in keys:
        ref = None
        for d in models:
            t = d.get(k)
            if isinstance(t, torch.Tensor):
                ref = t
                break
        if ref is None:
            continue
        tensors = []
        for d in models:
            t = d.get(k)
            if t is None:
                tensors.append(torch.zeros_like(ref, device=device))
            else:
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, device=device)
                tensors.append(t)
        result[k] = merge_func(tensors, weight_tensor, **kwargs)
    return result
