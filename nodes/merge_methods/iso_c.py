import torch

NODE_TYPE = 'merge_methods/iso_c'
NODE_CATEGORY = 'Merge method'


def _iso_c_merge(tensors, out_dtype):
    """Apply Iso-C merge to a list of tensors."""
    summed = sum(tensors)
    shape = summed.shape
    if len(shape) < 2:
        return (summed / len(tensors)).to(out_dtype)

    mat = summed.to(torch.float32).reshape(shape[0], -1)
    u, s, v = torch.linalg.svd(mat, full_matrices=False)
    iso = s.mean()
    merged = iso * (u @ v)
    return merged.reshape(shape).to(out_dtype)


def execute(node, inputs):
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
            result[k] = _iso_c_merge(tensors, torch_dtype)
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
        'properties': {},
        'tooltip': 'Isotropic merging in common subspace',
    }
