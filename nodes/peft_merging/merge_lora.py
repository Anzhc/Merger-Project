from ..utils import get_params
from utils.lora_type import load_lora_for_models
import torch

NODE_TYPE = 'peft_merging/merge_lora'
NODE_CATEGORY = 'PEFT merging'


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

    if isinstance(unet, dict):
        unet_sd = unet.get('data')
        unet_fmt = unet.get('format', 'pt')
        unet_dtype = unet.get('dtype')
    else:
        unet_sd = unet
        unet_fmt = 'pt'
        unet_dtype = None

    if isinstance(clip_l, dict):
        clip_l_sd = clip_l.get('data')
        clip_l_fmt = clip_l.get('format', 'pt')
        clip_l_dtype = clip_l.get('dtype')
    else:
        clip_l_sd = clip_l
        clip_l_fmt = 'pt'
        clip_l_dtype = None

    if isinstance(clip_g, dict):
        clip_g_sd = clip_g.get('data')
        clip_g_fmt = clip_g.get('format', 'pt')
        clip_g_dtype = clip_g.get('dtype')
    else:
        clip_g_sd = clip_g
        clip_g_fmt = 'pt'
        clip_g_dtype = None

    for idx, lora in enumerate(loras):
        filename = f"lora_{idx}"
        unet_sd, clip_l_sd, clip_g_sd = load_lora_for_models(
            unet_sd if merge_unet else None,
            clip_l_sd if merge_clip_l else None,
            clip_g_sd if merge_clip_g else None,
            lora,
            strength_unet,
            strength_clip_l,
            strength_clip_g,
            filename=filename,
            model_type=model_type,
            dtype=inter_dtype,
        )

    outputs = []
    if unet is not None:
        outputs.append({'data': unet_sd, 'format': unet_fmt, 'dtype': unet_dtype})
    else:
        outputs.append(None)

    if clip_l is not None:
        outputs.append({'data': clip_l_sd, 'format': clip_l_fmt, 'dtype': clip_l_dtype})
    else:
        outputs.append(None)

    if clip_g is not None:
        outputs.append({'data': clip_g_sd, 'format': clip_g_fmt, 'dtype': clip_g_dtype})
    else:
        outputs.append(None)

    return tuple(outputs)


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
