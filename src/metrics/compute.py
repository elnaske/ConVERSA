from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from espnet2.bin.gan_codec_inference import AudioCoding
import torch

def get_FLOPs(module: torch.nn.Module, inp: torch.Tensor) -> float:
    while len(inp.shape) < 3:
        inp = inp.unsqueeze(0)
    FLOPs = FlopCountAnalysis(module, inp)
    return FLOPs.total()

def get_MACs(module: torch.nn.Module, inp: torch.Tensor):
    macs, _ = get_model_complexity_info(module, tuple(inp.size()), as_strings=False, print_per_layer_stat=False)
    return macs

def get_n_operations(model: AudioCoding, inp: torch.Tensor):
    modules = {
        "total": model.model.codec,
        "encoder": model.model.codec.generator.encoder,
        "decoder": model.model.codec.generator.decoder,
    }
    quantizer = model.model.codec.generator.quantizer

    n_ops = {"MACs": {k: {} for k in modules.keys()}}

    enc_macs = get_MACs(modules["encoder"], inp)

    codes = model(inp)["codes"]
    codes_q = quantizer.decode(codes).squeeze()
    dec_macs = get_MACs(modules["decoder"], codes_q)

    n_ops["MACs"]["encoder"] = enc_macs
    n_ops["MACs"]["decoder"] = dec_macs
    n_ops["MACs"]["total"] = enc_macs + dec_macs

    return n_ops