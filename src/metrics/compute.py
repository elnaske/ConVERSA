from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from espnet2.bin.gan_codec_inference import AudioCoding
import torch

# def get_FLOPs(module: torch.nn.Module, inp: torch.Tensor) -> float:
#     FLOPs = FlopCountAnalysis(module, inp)
#     return FLOPs.total()

def get_FLOPs(audio_coding: AudioCoding) -> dict:
    """
    Calculates total number of floating point operations (FLOPs) per input sample.
    """
    codec = audio_coding.model.codec.generator
    enc = audio_coding.model.codec.generator.encoder
    dec = audio_coding.model.codec.generator.decoder
    q = audio_coding.model.codec.generator.quantizer

    fs = audio_coding.fs

    sample = torch.rand(1, 1, fs) # one second of audio
    total_flops = FlopCountAnalysis(codec, sample).total() / fs # FLOPs per input sample (dividing instead of using sample with shape (1,1,1) because that might be inaccurate for recurrent architectures)

    enc_flops = FlopCountAnalysis(enc, sample).total() / fs

    codes = audio_coding(audio=sample.squeeze(), encode_only=True)["codes"]
    codes_q = q.decode(codes)
    dec_flops = FlopCountAnalysis(dec, codes_q).total() / fs

    flops = {
        "total": total_flops,
        "encoder": enc_flops,
        "decoder": dec_flops,
    }

    return flops

def get_MACs(module: torch.nn.Module, inp: torch.Tensor):
    macs, _ = get_model_complexity_info(module, tuple(inp.size()), as_strings=False, print_per_layer_stat=False)
    return macs

# def get_n_operations(audio_coding: AudioCoding, inp: torch.Tensor):
#     modules = {
#         "encoder": audio_coding.model.codec.generator.encoder,
#         "decoder": audio_coding.model.codec.generator.decoder,
#     }
#     quantizer = audio_coding.model.codec.generator.quantizer

#     n_ops = {"MACs": {k: {} for k in modules.keys()}}

#     enc_macs = get_MACs(modules["encoder"], inp)

#     codes = audio_coding(inp, encode_only=True)["codes"]
#     codes_q = quantizer.decode(codes).squeeze()
#     dec_macs = get_MACs(modules["decoder"], codes_q)

#     n_ops["MACs"]["encoder"] = enc_macs
#     n_ops["MACs"]["decoder"] = dec_macs
#     n_ops["MACs"]["total"] = enc_macs + dec_macs

#     return n_ops