import torch
from ..utils import get_params

NODE_TYPE = 'merge_methods/iso_cts'
NODE_CATEGORY = 'Merge method'


def _iso_cts_merge(tensors, k, dtype):
    """Apply Iso-CTS merge to list of tensors."""
    summed = sum(tensors)
    shape = summed.shape
    mat = summed.to(torch.float32).reshape(shape[0], -1)
    m, n = mat.shape
    r = min(m, n)
    u, s, v = torch.linalg.svd(mat, full_matrices=False)
    k = max(0, min(int(k), r))
    if k == 0:
        # fall back to Iso-C
        iso = s.mean()
        merged = iso * (u @ v)
        return merged.reshape(shape).to(dtype)
    u_cm = u[:, :k]
    v_cm = v[:k, :]
    sigma_cm = s[:k]
    sum_sigma = sigma_cm.sum()
    comps_u = [u_cm]
    comps_v = [v_cm]
    s_list = []
    remaining = r - k
    t = len(tensors)
    s_per = remaining // t if t else 0
    if s_per > 0:
        for tmat in tensors:
            tm = tmat.to(torch.float32).reshape(shape[0], -1)
            resid = tm - u_cm @ (u_cm.T @ tm)
            ru, rs, rv = torch.linalg.svd(resid, full_matrices=False)
            comps_u.append(ru[:, :s_per])
            comps_v.append(rv[:s_per, :])
            s_list.append(rs[:s_per])
        sum_sigma += torch.cat(s_list).sum() if s_list else 0
    U_star = torch.cat(comps_u, dim=1)
    V_star = torch.cat(comps_v, dim=0)
    U_star, _ = torch.linalg.qr(U_star, mode='reduced')
    V_star_t, _ = torch.linalg.qr(V_star.T, mode='reduced')
    V_star = V_star_t.T
    r_final = U_star.shape[1]
    iso = sum_sigma / r_final
    merged = iso * (U_star @ V_star)
    return merged.reshape(shape).to(dtype)


def execute(node, inputs):
    params = get_params(node)
    k = float(params.get('k', 1))
    if len(inputs) < 2:
        raise ValueError('Iso-CTS requires at least two input models')
    dicts = []
    dtype = None
    fmt = 'pt'
    for inp in inputs:
        if isinstance(inp, dict):
            dicts.append(inp.get('data'))
            dtype = dtype or inp.get('dtype')
            fmt = inp.get('format', fmt)
        else:
            dicts.append(inp)
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = {}
    for kname in keys:
        ref = None
        tensors = []
        for d in dicts:
            t = d.get(kname)
            if t is None:
                if ref is None:
                    for other in dicts:
                        if kname in other:
                            ref = torch.as_tensor(other[kname])
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
            result[kname] = _iso_cts_merge(tensors, k, dtype or torch.float32)
    return {'data': result, 'format': fmt, 'dtype': dtype}


def get_spec():
    inputs = [{'name': chr(ord('A') + i), 'type': 'model'} for i in range(10)]
    return {
        'type': NODE_TYPE,
        'title': 'Iso-CTS Merge',
        'category': 'merge_methods',
        'inputs': inputs,
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {'kind': 'number', 'name': 'k', 'bind': 'k', 'options': {'min': 0, 'step': 1}},
        ],
        'properties': {'k': 1},
        'tooltip': 'Iso-CTS merge with common and task-specific subspaces',
    }
