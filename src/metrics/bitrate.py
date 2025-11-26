from espnet2.bin.gan_codec_inference import AudioCoding
import torch
import math

def get_codebook_size(model: AudioCoding) -> int:
    return model.model.codec.generator.quantizer.bins

def get_bitrate_kbps(model: AudioCoding, inp: torch.Tensor, fs: int, n_q=None) -> float:
    if n_q is None:
        n_q = model.model.codec.generator.quantizer.n_q

    bins = get_codebook_size(model)

    t = inp.size(-1) / fs
    codes = model(inp)["codes"]

    codes_per_sec = codes.size(-1) / t

    return (codes_per_sec * math.log2(bins) * n_q) / 1000
