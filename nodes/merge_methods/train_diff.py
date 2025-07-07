from ..utils import get_params
import torch

NODE_TYPE = 'merge_methods/train_diff'
NODE_CATEGORY = 'Merge method'


def _train_diff(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float, multiplier: float) -> torch.Tensor:
    if torch.allclose(b.float(), c.float(), rtol=0, atol=0):
        return torch.zeros_like(a)

    diff_ab = b.float() - c.float()
    distance_a0 = torch.abs(b.float() - c.float())
    distance_a1 = torch.abs(b.float() - a.float())
    sum_distances = distance_a0 + distance_a1

    base_scale = torch.where(sum_distances != 0,
                             distance_a1 / sum_distances,
                             torch.tensor(0., dtype=torch.float32, device=a.device))
    sign_scale = torch.sign(b.float() - c.float())
    scale = sign_scale * torch.abs(base_scale) * alpha

    new_diff = scale * torch.abs(diff_ab)
    return new_diff.to(a.dtype) * multiplier


def execute(node, inputs):
    params = get_params(node)
    alpha = float(params.get('alpha', 1.0))
    multiplier = float(params.get('multiplier', 1.8))
    if len(inputs) < 3:
        raise ValueError('TrainDiff merge requires three inputs')
    m1, m2, m3 = inputs[0], inputs[1], inputs[2]
    d1 = m1['data'] if isinstance(m1, dict) else m1
    d2 = m2['data'] if isinstance(m2, dict) else m2
    d3 = m3['data'] if isinstance(m3, dict) else m3
    dtype = m1.get('dtype') if isinstance(m1, dict) else None
    result = {}
    keys = set(d1.keys()) | set(d2.keys()) | set(d3.keys())
    for k in keys:
        v1 = d1.get(k, 0)
        v2 = d2.get(k, 0)
        v3 = d3.get(k, 0)
        t1 = v1 if isinstance(v1, torch.Tensor) else torch.tensor(v1)
        t2 = v2 if isinstance(v2, torch.Tensor) else torch.tensor(v2)
        t3 = v3 if isinstance(v3, torch.Tensor) else torch.tensor(v3)
        merged = t1 + _train_diff(t1, t2, t3, alpha=alpha, multiplier=multiplier)
        result[k] = merged
    fmt = m1.get('format', 'pt') if isinstance(m1, dict) else 'pt'
    return {'data': result, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Train Diff Merge',
        'category': 'merge_methods',
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
            {'name': 'C', 'type': 'model'},
        ],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {
                'kind': 'slider',
                'name': 'Alpha',
                'bind': 'alpha',
                'options': {'min': -1, 'max': 1, 'step': 0.01},
            },
            {
                'kind': 'slider',
                'name': 'Multiplier',
                'bind': 'multiplier',
                'options': {'min': 1, 'max': 3, 'step': 0.01},
            },
        ],
        'properties': {'alpha': 1.0, 'multiplier': 1.8},
        'tooltip': 'A + train_diff(A, B, C) * alpha * multiplier',
    }
