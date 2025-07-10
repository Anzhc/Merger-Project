from ..utils import get_params
from utils.lora_type import load_lora, model_lora_keys_unet, model_lora_keys_clip, calculate_weight
import torch

NODE_TYPE = 'peft_merging/merge_lora'
NODE_CATEGORY = 'PEFT merging'

def _merge_model(model_state, lora_states, key_map_func, model_type, dtype, strength):
    if model_state is None:
        return None
    key_map = key_map_func(model_state, {}, model_type) if model_type is not None else key_map_func(model_state, {})
    patches = {}
    for lora in lora_states:
        patch_dict = load_lora(lora, key_map, log_missing=False)
        for k, v in patch_dict.items():
            patches.setdefault(k, []).append((strength, v, 1.0, None, None))
    for k, plist in patches.items():
        if k in model_state:
            model_state[k] = calculate_weight(plist, model_state[k], k, intermediate_dtype=dtype)
    return model_state

def execute(node, inputs):
    params = get_params(node)
    dtype_name = params.get('dtype', 'fp16')
    dtype_map = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    inter_dtype = dtype_map.get(dtype_name, torch.float16)
    merge_unet = params.get('merge_unet', True)
    merge_clip_l = params.get('merge_clip_l', True)
    merge_clip_g = params.get('merge_clip_g', True)
    model_type = params.get('model_type', None)
    strength_unet = float(params.get('strength_unet', 1.0))
    strength_clip_l = float(params.get('strength_clip_l', 1.0))
    strength_clip_g = float(params.get('strength_clip_g', 1.0))

    unet = inputs[0] if len(inputs) > 0 else None
    clip_l = inputs[1] if len(inputs) > 1 else None
    clip_g = inputs[2] if len(inputs) > 2 else None
    loras = [x['data'] if isinstance(x, dict) else x for x in inputs[3:] if x is not None]

    out = []
    if unet is not None:
        sd = unet['data'] if isinstance(unet, dict) else unet
        if merge_unet:
            sd = _merge_model(sd, loras, model_lora_keys_unet, model_type, inter_dtype, strength_unet)
        out.append({'data': sd, 'format': unet.get('format', 'pt'), 'dtype': unet.get('dtype')})
    else:
        out.append(None)

    if clip_l is not None:
        sd = clip_l['data'] if isinstance(clip_l, dict) else clip_l
        if merge_clip_l:
            sd = _merge_model(sd, loras, model_lora_keys_clip, None, inter_dtype, strength_clip_l)
        out.append({'data': sd, 'format': clip_l.get('format', 'pt'), 'dtype': clip_l.get('dtype')})
    else:
        out.append(None)

    if clip_g is not None:
        sd = clip_g['data'] if isinstance(clip_g, dict) else clip_g
        if merge_clip_g:
            sd = _merge_model(sd, loras, model_lora_keys_clip, None, inter_dtype, strength_clip_g)
        out.append({'data': sd, 'format': clip_g.get('format', 'pt'), 'dtype': clip_g.get('dtype')})
    else:
        out.append(None)
    return tuple(out)


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Merge LoRA',
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
            {'kind': 'combo', 'name': 'DType', 'bind': 'dtype', 'options': {'values': ['fp32', 'fp16', 'bf16']}},
            {'kind': 'checkbox', 'name': 'Merge Unet', 'bind': 'merge_unet'},
            {'kind': 'checkbox', 'name': 'Merge Clip L', 'bind': 'merge_clip_l'},
            {'kind': 'checkbox', 'name': 'Merge Clip G', 'bind': 'merge_clip_g'},
            {'kind': 'slider', 'name': 'Unet Strength', 'bind': 'strength_unet', 'options': {'min': 0, 'max': 1, 'step': 0.01}},
            {'kind': 'slider', 'name': 'Clip L Strength', 'bind': 'strength_clip_l', 'options': {'min': 0, 'max': 1, 'step': 0.01}},
            {'kind': 'slider', 'name': 'Clip G Strength', 'bind': 'strength_clip_g', 'options': {'min': 0, 'max': 1, 'step': 0.01}},
            {'kind': 'combo', 'name': 'Model Type', 'bind': 'model_type', 'options': {'values': ['cascade', 'sd3', 'auraflow', 'pixart', 'hunyuan_dit', 'flux', 'genmo_mochi', 'hunyuan_video', 'base']}},
        ],
        'properties': {
            'dtype': 'fp16',
            'merge_unet': True,
            'merge_clip_l': True,
            'merge_clip_g': True,
            'strength_unet': 1.0,
            'strength_clip_l': 1.0,
            'strength_clip_g': 1.0,
            'model_type': 'base'
        },
        'tooltip': 'Merge multiple LoRA models into supplied UNet and CLIP checkpoints.'
    }
