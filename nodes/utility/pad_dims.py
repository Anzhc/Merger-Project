import torch

NODE_TYPE = 'utility/pad_dims'
NODE_CATEGORY = 'Utility'


def execute(node, inputs):
    if len(inputs) < 2:
        raise ValueError('Pad Dimensions requires two input models')
    m1, m2 = inputs[0], inputs[1]
    d1 = m1['data'] if isinstance(m1, dict) else m1
    d2 = m2['data'] if isinstance(m2, dict) else m2
    dtype = m1.get('dtype') if isinstance(m1, dict) else None
    result = {}
    for k, v1 in d1.items():
        v2 = d2.get(k)
        if (
            isinstance(v1, torch.Tensor)
            and isinstance(v2, torch.Tensor)
            and v1.shape != v2.shape
            and len(v1.shape) == len(v2.shape)
            and all(s1 <= s2 for s1, s2 in zip(v1.shape, v2.shape))
        ):
            padded = torch.zeros(v2.shape, dtype=v1.dtype, device=v1.device)
            slices = tuple(slice(0, s) for s in v1.shape)
            padded[slices] = v1
            result[k] = padded
        else:
            result[k] = v1
    fmt = m1.get('format', 'pt') if isinstance(m1, dict) else 'pt'
    return {'data': result, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Pad Dimensions',
        'category': 'utility',
        'node_category': NODE_CATEGORY,
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
        ],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [],
        'properties': {},
        'tooltip': 'Extend tensors in A with zeros to match B',
    }
