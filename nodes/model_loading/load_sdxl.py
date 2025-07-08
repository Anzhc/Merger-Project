import os
import traceback
import torch
from diffusers import StableDiffusionXLPipeline
from ..utils import get_params, get_device

NODE_TYPE = 'model_loading/load_sdxl'
NODE_CATEGORY = 'Model Loading'


def load_sdxl_submodules(checkpoint_file: str,
                          device: torch.device = None,
                          torch_dtype: torch.dtype = None):
    """Load SDXL checkpoint and return its main submodules."""
    print(f"Loading SDXL checkpoint from single file: {checkpoint_file}")
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_file)
    except Exception:
        print("Error loading SDXL pipeline from single file:")
        traceback.print_exc()
        return None

    unet = pipe.unet
    vae = pipe.vae
    clip_l = pipe.text_encoder
    clip_g = pipe.text_encoder_2

    if device is not None or torch_dtype is not None:
        for module in [unet, vae, clip_l, clip_g]:
            module.to(device=device, dtype=torch_dtype)

    # free pipeline
    del pipe
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return unet, clip_l, clip_g, vae


def execute(node, inputs):
    params = get_params(node)
    path = params.get('path')
    dtype = params.get('dtype', 'fp16')
    if not path:
        raise ValueError('No path provided for SDXL Loader')
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    device = torch.device(get_device(node))
    path = os.path.expanduser(os.path.expandvars(path))

    modules = load_sdxl_submodules(path, device=device, torch_dtype=torch_dtype)
    if modules is None:
        raise RuntimeError('Failed to load SDXL checkpoint')
    # return as a tuple matching the output order
    return modules


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Load SDXL',
        'category': 'model_loading',
        'inputs': [],
        'outputs': [
            {'name': 'unet', 'type': 'model'},
            {'name': 'clip_l', 'type': 'model'},
            {'name': 'clip_g', 'type': 'model'},
            {'name': 'vae', 'type': 'model'},
        ],
        'widgets': [
            {'kind': 'button', 'name': 'Choose File', 'action': '/choose_file', 'assignTo': 'path'},
            {'kind': 'text', 'name': 'File', 'bind': 'path', 'options': {'disabled': True}},
            {'kind': 'combo', 'name': 'DType', 'bind': 'dtype', 'options': {'values': ['fp32', 'fp16', 'bf16']}},
        ],
        'properties': {'path': '', 'dtype': 'fp16'},
        'tooltip': 'Load an SDXL checkpoint from safetensors',
    }
