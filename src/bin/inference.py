#!/usr/bin/env python3

# Pipeline for constrained codec evaluation
# Adapted from espnet2.bin.gan_codec_inference

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import soundfile as sf
import torch
from packaging.version import parse as V  # noqa
from typeguard import typechecked

import torch.quantization
from torch.profiler import profile, ProfilerActivity, record_function
from collections import defaultdict
from espnet2.bin.gan_codec_inference import AudioCoding
from torchaudio.transforms import Resample

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_codec.dac import DAC
from espnet2.gan_codec.soundstream import SoundStream
from espnet2.tasks.gan_codec import GANCodecTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import float_or_none, str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args

from src.utils import load_config

@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    target_bandwidth: Optional[float],
    encode_only: bool,
    always_fix_seed: bool,
    allow_variable_data_keys: bool,
    quantize_model: Optional[bool],
    quantize_modules: Optional[List[str]],
    quantize_dtype: Optional[str],
    benchmark_config: Optional[str],
):
    """Run speech coding inference."""
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    config = load_config(benchmark_config)
    n_q = config["n_codebooks"] if config["n_codebooks"] else -1

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    audio_coding_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        target_bandwidth=target_bandwidth,
        dtype=dtype,
        device=device,
        seed=seed,
        always_fix_seed=always_fix_seed,
        quantize_model=quantize_model,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
    )
    audio_coding = AudioCoding.from_pretrained(
        model_tag=model_tag,
        **audio_coding_kwargs,
    )

    # 3. Build data-iterator
    loader = GANCodecTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=GANCodecTask.build_preprocess_fn(audio_coding.train_args, False),
        collate_fn=GANCodecTask.build_collate_fn(audio_coding.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # Quick fix
    resample = Resample(24000, 16000)
    
    # Profiler activities
    activities = [ProfilerActivity.CPU]
    if ngpu:
        activities += [ProfilerActivity.CUDA]

    # 4. Start for-loop
    output_dir_path = Path(output_dir)
    (output_dir_path / "codec").mkdir(parents=True, exist_ok=True)
    (output_dir_path / "wav").mkdir(parents=True, exist_ok=True)
    (output_dir_path / "benchmark").mkdir(parents=True, exist_ok=True)

    
    with NpyScpWriter(
        output_dir_path / "codec", output_dir_path / "codec/codec.scp"
    ) as codec_writer:
        for i, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert _bs == 1, _bs

            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            batch["audio"] = resample(batch["audio"])

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            #     with record_function("total"):
            #         with record_function("encoder"):
            #             codes = audio_coding(**batch, encode_only=True)["codes"][:n_q]
            #         with record_function("decoder"):
            #             output_dict = audio_coding.decode(codes)
            # events = {evt.key: evt for evt in prof.key_averages() if evt.key in ["total", "encoder", "decoder"]}
            
            start_time = time.perf_counter()
            codes = audio_coding(**batch, encode_only=True)["codes"][:n_q]
            enc_latency = time.perf_counter() - start_time

            start_time = time.perf_counter()
            output_dict = audio_coding.decode(codes)
            dec_latency = time.perf_counter() - start_time
            total_latency = enc_latency + dec_latency

            output_dict.update(codes=codes)

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            
            if output_dict.get("resyn_audio") is not None:
                wav = output_dict["resyn_audio"].squeeze()

                logging.info(
                    f"({i}/?) {key} finished."
                )
                sf.write(
                    f"{output_dir_path}/wav/{key}.wav",
                    wav.cpu().numpy(),
                    audio_coding.fs,
                    "PCM_16",
                )

            if output_dict.get("codes") is not None:
                codec_writer[key] = output_dict["codes"].cpu().numpy()

            benchmark = defaultdict(dict)
            benchmark["key"] = key

            # for module, evt in events.items():
            #     latency = evt.cpu_time / 1e6 # in seconds
            #     benchmark[module]["latency"] = latency 
            #     benchmark[module]["samples_per_second"] = insize / (latency + 1e-6)
            #     benchmark[module]["RTF"] = (insize / audio_coding.fs) / (latency + 1e-6)
            #     benchmark[module]["CPU_mem"] = evt.cpu_memory_usage

            for module, latency in [("total", total_latency), ("encoder", enc_latency), ("decoder", dec_latency)]:
                benchmark[module]["latency"] = latency 
                benchmark[module]["samples_per_second"] = insize / (latency + 1e-6)
                benchmark[module]["RTF"] = (insize / audio_coding.fs) / (latency + 1e-6)

            with open((output_dir_path / "benchmark" / "results.txt"), "a") as f:
                f.write(json.dumps(benchmark)+"\n")

    # remove files if those are not included in output dict
    if output_dict.get("codes") is None:
        shutil.rmtree(output_dir_path / "codes")
    if output_dict.get("resyn_audio") is None:
        shutil.rmtree(output_dir_path / "wav")

def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="Codec inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    parser.add_argument(
        "--benchmark_config",
        type=str,
        required=True,
        help="Config file for benchmark"
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )
    group.add_argument(
        "--target_bandwidth",
        type=float_or_none,
        default=None,
        help="Target bandwidth for models supporting various bandwidth",
    )
    group.add_argument(
        "--encode_only",
        type=str2bool,
        default=False,
        help="Whether to only do encoding.",
    )
    group.add_argument(
        "--always_fix_seed",
        type=str2bool,
        default=False,
        help="Whether to always fix seed",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to the model.",
    )
    group.add_argument(
        "--quantize_modules",
        type=str,
        nargs="*",
        default=["Linear"],
        help="""List of modules to be dynamically quantized.
        E.g.: --quantize_modules=[Linear,LSTM,GRU].
        Each specified module should be an attribute of 'torch.nn', e.g.:
        torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, ...""",
    )
    group.add_argument(
        "--quantize_dtype",
        type=str,
        default="qint8",
        choices=["float16", "qint8"],
        help="Dtype for dynamic quantization.",
    )

    return parser


def main(cmd=None):
    """Run Codec model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
