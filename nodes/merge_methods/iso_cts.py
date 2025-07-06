import torch
from ..utils import get_params

NODE_TYPE = 'merge_methods/iso_cts'
NODE_CATEGORY = 'Merge method'


def _iso_cts_merge(tensors, frac, out_dtype):
    """Return the Iso-CTS update matrix for a set of task deltas."""
    summed = sum(tensors)
    shape = summed.shape

    # Bias and embedding parameters are treated by simple averaging.
    if len(shape) < 2:
        return (summed / len(tensors)).to(out_dtype)

    mat = summed.to(torch.float32).reshape(shape[0], -1)
    m, n = mat.shape
    r = min(m, n)
    u, s, v = torch.linalg.svd(mat, full_matrices=False)

    # Size of the common subspace
    k = max(0, min(int(r * float(frac)), r))
    if k == 0:
        iso = s.mean()
        delta = iso * (u @ v)
        return delta.reshape(shape).to(out_dtype)

    u_cm = u[:, :k]
    v_cm = v[:k, :]
    sigma_cm = s[:k]
    sum_sigma = sigma_cm.sum()
    comps_u = [u_cm]
    comps_v = [v_cm]

    # Number of task-specific components per task
    remaining = r - k
    num_tasks = len(tensors)
    s_per_task = remaining // num_tasks if num_tasks else 0
    if s_per_task > 0:
        for tmat in tensors:
            tm = tmat.to(torch.float32).reshape(shape[0], -1)
            # Project onto the subspace orthogonal to the common one (Eq.10)
            resid = tm - u_cm @ (u_cm.T @ tm)
            ru, rs, rv = torch.linalg.svd(resid, full_matrices=False)
            comps_u.append(ru[:, :s_per_task])
            comps_v.append(rv[:s_per_task, :])
            sum_sigma += rs[:s_per_task].sum()

    U_star = torch.cat(comps_u, dim=1)
    V_star = torch.cat(comps_v, dim=0)

    # Whitening using SVD (Eq.11). Fall back to QR if SVD fails to converge.
    try:
        Pu, _, Qu = torch.linalg.svd(U_star, full_matrices=False)
        U_star = Pu @ Qu
    except Exception:
        U_star, _ = torch.linalg.qr(U_star, mode='reduced')

    try:
        Pv, _, Qv = torch.linalg.svd(V_star, full_matrices=False)
        V_star = Pv @ Qv
    except Exception:
        V_star_t, _ = torch.linalg.qr(V_star.T, mode='reduced')
        V_star = V_star_t.T

    r_final = U_star.shape[1]
    iso = sum_sigma / r_final
    delta = iso * (U_star @ V_star)
    return delta.reshape(shape).to(out_dtype)


def execute(node, inputs):
    params = get_params(node)
    frac = float(params.get('fraction', 0.8))
    if len(inputs) < 2:
        raise ValueError('Iso-CTS requires at least two input models')
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
            result[kname] = _iso_cts_merge(tensors, frac, torch_dtype)
    return {'data': result, 'format': fmt, 'dtype': dtype_str}


def get_spec():
    inputs = [{'name': chr(ord('A') + i), 'type': 'model'} for i in range(10)]
    return {
        'type': NODE_TYPE,
        'title': 'Iso-CTS Merge',
        'category': 'merge_methods',
        'inputs': inputs,
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {
                'kind': 'slider',
                'name': 'Common fraction',
                'bind': 'fraction',
                'options': {'min': 0, 'max': 1, 'step': 0.05},
            },
        ],
        'properties': {'fraction': 0.8},
        'tooltip': 'Iso-CTS update using common and task-specific subspaces',
    }
