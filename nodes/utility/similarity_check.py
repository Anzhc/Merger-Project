from ..utils import get_params
import torch

NODE_TYPE = 'utility/similarity_check'
NODE_CATEGORY = 'Utility'


def _tensor(v):
    if isinstance(v, torch.Tensor):
        return v.float().view(-1)
    return torch.tensor(v, dtype=torch.float32).view(-1)


def _jaccard(d1, d2):
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    if not k1 and not k2:
        return 100.0
    return 100.0 * len(k1 & k2) / len(k1 | k2)


def _magnitude(d1, d2):
    keys = set(d1.keys()) & set(d2.keys())
    if not keys:
        return 0.0
    diffs = []
    for k in keys:
        t1 = _tensor(d1[k])
        t2 = _tensor(d2[k])
        denom = torch.sum(torch.abs(t1))
        if denom == 0:
            denom = torch.sum(torch.abs(t2))
        if denom == 0:
            continue
        diff = torch.sum(torch.abs(t1 - t2)) / denom
        diffs.append(diff.item())
    if not diffs:
        return 0.0
    avg = sum(diffs) / len(diffs)
    sim = max(0.0, 1 - avg)
    return sim * 100


def _cosine(d1, d2):
    keys = set(d1.keys()) & set(d2.keys())
    if not keys:
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for k in keys:
        t1 = _tensor(d1[k])
        t2 = _tensor(d2[k])
        dot += torch.dot(t1, t2).item()
        norm1 += torch.dot(t1, t1).item()
        norm2 += torch.dot(t2, t2).item()
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos = dot / ((norm1 ** 0.5) * (norm2 ** 0.5))
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1) * 50


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 2:
        raise ValueError('Similarity check requires two input models')
    m1, m2 = inputs[0], inputs[1]
    d1 = m1['data'] if isinstance(m1, dict) else m1
    d2 = m2['data'] if isinstance(m2, dict) else m2
    algo = params.get('algorithm', 'jaccard')
    if algo == 'magnitude':
        score = _magnitude(d1, d2)
    elif algo == 'cosine':
        score = _cosine(d1, d2)
    else:
        score = _jaccard(d1, d2)
    return score


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Similarity Check',
        'category': 'utility',
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
        ],
        'outputs': [{'name': 'score', 'type': 'number'}],
        'widgets': [
            {
                'kind': 'combo',
                'name': 'Algorithm',
                'bind': 'algorithm',
                'options': {'values': ['jaccard', 'magnitude', 'cosine']},
            }
        ],
        'properties': {'algorithm': 'jaccard'},
        'tooltip': 'Compute similarity between two models (0-100%)',
    }
