from ..utils import get_params
from utils.peft_utils import parse_weights
from utils.lora_merge import merge_lora_models
import copy

NODE_TYPE = 'peft_merging/apply_lora'
NODE_CATEGORY = 'PEFT merging'


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 3:
        raise ValueError('Apply LoRA requires unet, clip_l and clip_g inputs')

    # base modules
    unet = inputs[0]['data'] if isinstance(inputs[0], dict) else inputs[0]
    clip_l = inputs[1]['data'] if isinstance(inputs[1], dict) else inputs[1]
    clip_g = inputs[2]['data'] if isinstance(inputs[2], dict) else inputs[2]

    # copy to avoid modifying originals
    unet = copy.deepcopy(unet) if unet is not None else None
    clip_l = copy.deepcopy(clip_l) if clip_l is not None else None
    clip_g = copy.deepcopy(clip_g) if clip_g is not None else None

    loras = []
    for inp in inputs[3:]:
        if inp is None:
            continue
        loras.append(inp['data'] if isinstance(inp, dict) else inp)

    if not loras:
        # nothing to merge
        return inputs[0], inputs[1], inputs[2]

    weights = parse_weights(params.get('weights', ''), len(loras))

    merge_unet = bool(params.get('merge_unet', True))
    merge_clip_l = bool(params.get('merge_clip_l', True))
    merge_clip_g = bool(params.get('merge_clip_g', True))

    merge_lora_models(unet, clip_l, clip_g, loras, weights,
                      merge_unet=merge_unet,
                      merge_clip_l=merge_clip_l,
                      merge_clip_g=merge_clip_g)

    def pack(sd, ref):
        if isinstance(ref, dict):
            fmt = ref.get('format', 'pt')
            dtype = ref.get('dtype')
            return {'data': sd, 'format': fmt, 'dtype': dtype}
        return sd

    return pack(unet, inputs[0]), pack(clip_l, inputs[1]), pack(clip_g, inputs[2])


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Apply LoRA',
        'category': 'peft_merging',
        'inputs': [
            {'name': 'unet', 'type': 'model'},
            {'name': 'clip_l', 'type': 'model'},
            {'name': 'clip_g', 'type': 'model'},
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
            {'name': 'C', 'type': 'model'},
            {'name': 'D', 'type': 'model'},
            {'name': 'E', 'type': 'model'},
            {'name': 'F', 'type': 'model'},
            {'name': 'G', 'type': 'model'},
            {'name': 'H', 'type': 'model'},
            {'name': 'I', 'type': 'model'},
            {'name': 'J', 'type': 'model'},
        ],
        'outputs': [
            {'name': 'unet', 'type': 'model'},
            {'name': 'clip_l', 'type': 'model'},
            {'name': 'clip_g', 'type': 'model'},
        ],
        'widgets': [
            {'kind': 'text', 'name': 'Weights', 'bind': 'weights'},
            {'kind': 'checkbox', 'name': 'Unet', 'bind': 'merge_unet'},
            {'kind': 'checkbox', 'name': 'CLIP-L', 'bind': 'merge_clip_l'},
            {'kind': 'checkbox', 'name': 'CLIP-G', 'bind': 'merge_clip_g'},
        ],
        'properties': {
            'weights': '',
            'merge_unet': True,
            'merge_clip_l': True,
            'merge_clip_g': True,
        },
        'tooltip': 'Merge one or more LoRA models into the provided checkpoint weights',
    }
