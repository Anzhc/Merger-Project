from ..utils import get_params
import device_manager
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

NODE_TYPE = 'merge_methods/pleas_merge'
NODE_CATEGORY = 'Merge method'


def _merge_tensors(t1: torch.Tensor, t2: torch.Tensor, samples: int, device: torch.device) -> torch.Tensor:
    if t1.shape != t2.shape or t1.dim() < 2:
        return (t1 + t2) / 2
    out_dim = t1.shape[0]
    t1_flat = t1.reshape(out_dim, -1).float()
    t2_flat = t2.reshape(out_dim, -1).float()
    cost = torch.cdist(t1_flat, t2_flat).cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    t2_perm = t2_flat[col_ind]
    X = torch.randn(samples, t1_flat.shape[1], device=device)
    Y = 0.5 * (X @ t1_flat.T + X @ t2_perm.T)
    sol = torch.linalg.lstsq(X, Y).solution.T
    return sol.reshape_as(t1).to(t1.dtype)


def _pleas_merge_dicts(a: dict, b: dict, samples: int) -> dict:
    device = torch.device(device_manager.get_device())
    keys = set(a.keys()) & set(b.keys())
    result = {}
    for k in tqdm(keys, desc='PLeaS merge', unit='param'):
        t1, t2 = a[k], b[k]
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1, device=device)
        else:
            t1 = t1.to(device)
        if not isinstance(t2, torch.Tensor):
            t2 = torch.tensor(t2, device=device)
        else:
            t2 = t2.to(device)
        result[k] = _merge_tensors(t1, t2, samples=samples, device=device)
    return result


def execute(node, inputs):
    params = get_params(node)
    samples = int(params.get('samples', 256))
    if len(inputs) < 2:
        raise ValueError('PLeaS merge requires two inputs')
    m1, m2 = inputs[0], inputs[1]
    d1 = m1['data'] if isinstance(m1, dict) else m1
    d2 = m2['data'] if isinstance(m2, dict) else m2
    dtype = m1.get('dtype') if isinstance(m1, dict) else None
    fmt = m1.get('format', 'pt') if isinstance(m1, dict) else 'pt'
    merged = _pleas_merge_dicts(d1, d2, samples=samples)
    return {'data': merged, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'PLeaS Merge',
        'category': 'merge_methods',
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
        ],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {'kind': 'number', 'name': 'Samples', 'bind': 'samples', 'options': {'min': 32, 'max': 2048}},
        ],
        'properties': {'samples': 256},
        'tooltip': 'Merge two models via permutations and least squares',
    }
