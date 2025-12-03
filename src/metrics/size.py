import os
import torch
from espnet2.bin.gan_codec_inference import AudioCoding
from typing import Dict
import uuid

def get_n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def get_size_mb(model: torch.nn.Module) -> int:
    temp = f"temp_{uuid.uuid4()}.pth"
    torch.save(model.state_dict(), temp)
    size_mb = os.path.getsize(temp) / (1024 * 1024)
    os.remove(temp)
    return size_mb

def get_model_size(model: AudioCoding) -> Dict[str, Dict[str, int]]:
    """
    Calculates number of parameters and file size of the model.

    Args:
        model: ESPNet GAN codec model

    Returns:
        Dict containing model size metrics for each block.
    """
    modules = {
        "total": model.model.codec,
        "generator": model.model.codec.generator,
        "encoder": model.model.codec.generator.encoder,
        "decoder": model.model.codec.generator.decoder,
        "discriminator": model.model.codec.discriminator,
    }

    model_size = {metric: {k: {} for k in modules.keys()} for metric in ["n_params", "size_mb"]}
    for name, module in modules.items():
        model_size["n_params"][name] = get_n_params(module)
        model_size["size_mb"][name] = get_size_mb(module)
    
    return model_size