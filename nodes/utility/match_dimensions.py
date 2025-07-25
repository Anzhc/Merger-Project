import torch

NODE_TYPE = 'utility/match_dimensions'
NODE_CATEGORY = 'Utility'


def execute(node, inputs):
    if len([inp for inp in inputs if inp is not None]) < 2:
        raise ValueError('Match Dimensions requires at least two input models')

    models = []  # list of tuples (data, fmt, dtype)
    for inp in inputs:
        if inp is None:
            models.append(None)
            continue
        if isinstance(inp, dict):
            models.append((inp.get('data'), inp.get('format', 'pt'), inp.get('dtype')))
        else:
            models.append((inp, 'pt', None))

    # gather shapes for keys present in multiple models
    shape_counts = {}
    shapes = {}
    for info in models:
        if info is None:
            continue
        data = info[0]
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                continue
            shapes.setdefault(k, []).append(v.shape)
            shape_counts[k] = shape_counts.get(k, 0) + 1

    pad_shapes = {}
    for k, shape_list in shapes.items():
        if shape_counts.get(k, 0) < 2:
            continue
        dims_lens = {len(s) for s in shape_list}
        if len(dims_lens) != 1:
            continue
        rank = dims_lens.pop()
        max_shape = [0] * rank
        for s in shape_list:
            for i, dim in enumerate(s):
                if dim > max_shape[i]:
                    max_shape[i] = dim
        pad_shapes[k] = tuple(max_shape)

    results = []
    for info in models:
        if info is None:
            results.append(None)
            continue
        data, fmt, dtype = info
        if not isinstance(data, dict):
            results.append({'data': data, 'format': fmt, 'dtype': dtype})
            continue
        new_data = {}
        for k, v in data.items():
            target = pad_shapes.get(k)
            if target and isinstance(v, torch.Tensor):
                if len(v.shape) == len(target) and all(s <= t for s, t in zip(v.shape, target)):
                    if v.shape != target:
                        padded = torch.zeros(target, dtype=v.dtype, device=v.device)
                        slices = tuple(slice(0, s) for s in v.shape)
                        padded[slices] = v
                        new_data[k] = padded
                    else:
                        new_data[k] = v
                else:
                    new_data[k] = v
            else:
                new_data[k] = v
        results.append({'data': new_data, 'format': fmt, 'dtype': dtype})

    return tuple(results)


def get_spec():
    inputs = [{'name': chr(65 + i), 'type': 'model'} for i in range(10)]
    outputs = [{'name': chr(65 + i), 'type': 'model'} for i in range(10)]
    return {
        'type': NODE_TYPE,
        'title': 'Match Dimensions',
        'category': 'utility',
        'node_category': NODE_CATEGORY,
        'inputs': inputs,
        'outputs': outputs,
        'widgets': [],
        'properties': {},
        'tooltip': 'Pad tensors in each input model to match the largest dimensions across models',
    }

