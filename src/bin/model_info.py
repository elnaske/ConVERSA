#!/usr/bin/env python3

# Pipeline for constrained codec evaluation
# Adapted from espnet2.bin.gan_codec_inference

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

from packaging.version import parse as V  # noqa
from typeguard import typechecked

from collections import defaultdict
from espnet2.bin.gan_codec_inference import AudioCoding

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_codec.dac import DAC
from espnet2.gan_codec.soundstream import SoundStream
from espnet2.tasks.gan_codec import GANCodecTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import float_or_none, str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args

from src.metrics import get_model_size, get_bitrate_kbps, get_codebook_size, get_n_codebooks, get_FLOPs
from src.utils import load_config, save_to_json


@typechecked
def get_model_info(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    target_bandwidth: Optional[float],
    encode_only: bool,
    always_fix_seed: bool,
    quantize_model: bool = False,
    quantize_modules: List[str] = ["Linear"],
    quantize_dtype: str = "qint8",
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

    pipeline_config = load_config("../../../../config.yaml")

    benchmark = defaultdict(dict)

    benchmark["model_tag"] = model_tag
    benchmark["precision"] = quantize_dtype if quantize_model else "float32"
    benchmark["fs"] = audio_coding.fs
    benchmark["n_codebooks"] = pipeline_config["n_codebooks"] if pipeline_config["n_codebooks"] else get_n_codebooks(audio_coding)
    benchmark["codebook_size"] = get_codebook_size(audio_coding)
    benchmark["model_size"] = get_model_size(audio_coding)
    benchmark["kbps"] = get_bitrate_kbps(audio_coding)
    benchmark["FLOPs_per_sample"] = get_FLOPs(audio_coding)

    output_dir_path = Path(output_dir)
    (output_dir_path / "benchmark").mkdir(parents=True, exist_ok=True)
    save_to_json(benchmark, (output_dir_path / "benchmark" / "model_info.json"))


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
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    get_model_info(**kwargs)


if __name__ == "__main__":
    main()
