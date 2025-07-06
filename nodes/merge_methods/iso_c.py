import torch
from ..utils import get_params


def _safe_svd(mat, full_matrices=False):
    """Return SVD using the ``gesvd`` driver when available."""
    try:
        return torch.linalg.svd(mat, full_matrices=full_matrices, driver="gesvd")
    except Exception:
        return torch.linalg.svd(mat, full_matrices=full_matrices)

NODE_TYPE = 'merge_methods/iso_c'
NODE_CATEGORY = 'Merge method'


def _iso_c_merge(tensors, out_dtype, device):
    """Return the Iso-C update matrix given task deltas on ``device``."""
    summed = sum(t.to(device=device, dtype=torch.float32) for t in tensors)
    shape = summed.shape

    # For 1D parameters (biases, embeddings) the SVD reduces to
    # averaging, which matches the formulation in the paper.
    if len(shape) < 2:
        return (summed / len(tensors)).to(out_dtype)

    mat = summed.reshape(shape[0], -1)
    u, s, v = _safe_svd(mat, full_matrices=False)

    # Isotropic scaling factor (Eq.7)
    iso = s.mean()

    delta = iso * (u @ v)
    return delta.reshape(shape).to(out_dtype)


def execute(node, inputs):
    params = get_params(node)
    device_idx = int(params.get('cuda', 0))
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')

    if len(inputs) < 2:
        raise ValueError('Iso-C requires at least two input models')

    # collect parameter dictionaries
    dicts = []
    dtype_str = None
    fmt = 'pt'
    for inp in inputs:
        if isinstance(inp, dict):
            dicts.append(inp.get('data'))
            dtype_str = dtype_str or inp.get('dtype')
            fmt = inp.get('format', fmt)
        else:
            dicts.append(inp)

    keys = set()
    for d in dicts:
        keys.update(d.keys())

    # resolve dtype for computations
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float32)

    result = {}
    for k in keys:
        ref = None
        tensors = []
        for d in dicts:
            t = d.get(k)
            if t is None:
                if ref is None:
                    # find reference tensor shape
                    for other in dicts:
                        if k in other:
                            ref = torch.as_tensor(other[k])
                            break
                if ref is None:
                    continue
                t = torch.zeros_like(ref)
            else:
                t = torch.as_tensor(t)
                if ref is None:
                    ref = t
            tensors.append(t)
        if tensors:
            result[k] = _iso_c_merge(tensors, torch_dtype, device).to(torch_dtype).cpu()
        else:
            continue

    return {'data': result, 'format': fmt, 'dtype': dtype_str}


def get_spec():
    inputs = [{'name': chr(ord('A') + i), 'type': 'model'} for i in range(10)]
    return {
        'type': NODE_TYPE,
        'title': 'Iso-C Merge',
        'category': 'merge_methods',
        'inputs': inputs,
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {
                'kind': 'number',
                'name': 'CUDA device',
                'bind': 'cuda',
                'options': {'min': 0, 'step': 1},
            }
        ],
        'properties': {'cuda': 0},
        'tooltip': 'Compute Iso-C update (delta) in the common subspace',
    }
