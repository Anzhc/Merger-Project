from ..utils import get_params

NODE_TYPE = 'merge_methods/add_subtract'
NODE_CATEGORY = 'Merge method'


def execute(node, inputs):
    params = get_params(node)
    alpha = float(params.get('alpha', 1.0))
    if len(inputs) < 2:
        raise ValueError('Add/Subtract node requires two inputs')
    m1, m2 = inputs[0], inputs[1]
    d1 = m1['data'] if isinstance(m1, dict) else m1
    d2 = m2['data'] if isinstance(m2, dict) else m2
    dtype = m1.get('dtype') if isinstance(m1, dict) else None
    keys = set(d1.keys()) | set(d2.keys())
    result = {}
    for k in keys:
        v1 = d1.get(k, 0)
        v2 = d2.get(k, 0)
        result[k] = v1 + v2 * alpha
    fmt = m1.get('format', 'pt') if isinstance(m1, dict) else 'pt'
    return {'data': result, 'format': fmt, 'dtype': dtype}


def get_spec():
    """Return UI specification for this node."""
    return {
        'type': NODE_TYPE,
        'title': 'Add/Subtract',
        'category': 'merge_methods',
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
        ],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {
                'kind': 'slider',
                'name': 'Alpha',
                'bind': 'alpha',
                'options': {'min': -1, 'max': 1, 'step': 0.01},
            }
        ],
        'properties': {'alpha': 1.0},
        'tooltip': 'A + B * alpha',
    }
