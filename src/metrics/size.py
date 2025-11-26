import os
import torch
from typing import Dict

def get_n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def get_size_mb(model) -> int:
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size_mb

def get_model_size(model) -> Dict[str, Dict[str, int]]:
    """
    Calculates number of parameters and file size of the model.

    Args:
        model: ESPNet GAN codec model

    Returns:
        Dict containing model size metrics for each block.
    """
    models = {
        "total": model.codec,
        "generator": model.codec.generator,
        "encoder": model.codec.generator.encoder,
        "decoder": model.codec.generator.decoder,
        "discriminator": model.codec.discriminator,
    }

    model_size = {k: {} for k in models.keys()}
    for name, module in models.items():
        model_size[name]["n_params"] = get_n_params(module)
        model_size[name]["size_mb"] = get_size_mb(module)
    
    return model_size