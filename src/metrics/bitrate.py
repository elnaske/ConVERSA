from espnet2.bin.gan_codec_inference import AudioCoding
import torch
import math
from typing import Union

def get_codebook_size(audio_coding: AudioCoding) -> int:
    return audio_coding.model.codec.generator.quantizer.bins

def get_n_codebooks(audio_coding: AudioCoding) -> int:
    return audio_coding.model.codec.generator.quantizer.n_q

def get_bitrate_kbps(audio_coding: AudioCoding, n_q: Union[int, None] = None) -> float:
    if n_q is None:
        n_q = get_n_codebooks(audio_coding)
    bins = get_codebook_size(audio_coding)

    sample = torch.rand(audio_coding.fs) # one second of audio
    codes = audio_coding(audio=sample, encode_only=True)["codes"]
    codes_per_sec = codes.size(-1)
    
    return (codes_per_sec * math.log2(bins) * n_q) / 1000
