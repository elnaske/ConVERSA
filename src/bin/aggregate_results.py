import argparse
import json
import os
from collections import defaultdict
from glob import glob
import numpy as np

from src.utils import save_to_json

def load_versa_avg_results(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    
    out = {}
    for line in lines:
        metric, value = line.split(": ")
        out[metric] = float(value.strip())

    return out

def load_versa_utt_results(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    out = {}
    for line in lines:
        line = (
            line.strip()
            .replace("'", '"')
            .replace("inf", "Infinity")
            .replace(f"nan", "0.0")
        )
        res = json.loads(line)
        key = res.pop("key")
        out[key] = res

    return out

def load_model_info(logdir: str):
    with open(os.path.join(logdir, "log/benchmark/model_info.json")) as f:
        out = json.load(f)
    return out

def aggregate_benchmark(paths: list[str]) -> dict:
    out = {}
    for path in paths:
        with open(path, "r") as f:
            res = json.load(f)
        out.update(res)
    return out

def compute_FLOPS(benchmark: dict, model_info: float) -> dict:
    """
    Computes the number of floating point operations per second (FLOPS) from the benchmark results and the FLOPs computed during the model information step.
    """
    for key, modules in benchmark.items():
        for module, metrics in modules.items():
            FLOPS = model_info["FLOPs_per_sample"][module] * metrics["samples_per_second"]
            metrics.update(FLOPS=FLOPS)
    return benchmark

def get_benchmark_avgs(benchmark: dict) -> dict:
    """
    Computes average values across utterances for all benchmark metrics.
    """
    res = defaultdict(dict)
    for key in benchmark.keys():
        for module, metrics in benchmark[key].items():
            if not module in res.keys():
                res[module] = defaultdict(list)
            for metric, value in metrics.items():
                res[module][metric] += [value]
    avgs = {}
    for module, metrics in res.items():
        avgs[module] = {}
        for metric in metrics.keys():
            avgs[module][metric] = np.mean(res[module][metric]).item()

    return avgs

def get_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate results from inference benchmark and VERSA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Input log directory."
    )
    parser.add_argument(
        "--scoredir",
        type=str,
        required=True,
        help="Output scoring directory."
    )
    return parser

def aggregate_results(logdir: str, scoredir: str):
    res = defaultdict(dict)

    model_info = load_model_info(logdir)
    res["model_info"] = model_info

    benchmark_paths = glob(os.path.join(logdir, "log/output.*/benchmark/results.json"))
    benchmark = aggregate_benchmark(benchmark_paths)
    benchmark = compute_FLOPS(benchmark, model_info)

    res["averages"]["benchmark"] = get_benchmark_avgs(benchmark)
    res["utterances"]["benchmark"] = benchmark

    res["averages"]["VERSA"] = load_versa_avg_results(os.path.join(logdir, "score/avg_result.txt"))
    res["utterances"]["VERSA"] = load_versa_utt_results(os.path.join(logdir, "score/utt_result.txt"))

    save_to_json(res, os.path.join(scoredir, "results.json"))


def main():
    parser = get_parser()
    args = parser.parse_args()
    aggregate_results(args.logdir, args.scoredir)

if __name__=="__main__":
    main()