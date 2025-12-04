#!/usr/bin/env bash

set -e
set -u
set -o pipefail

CWD=$(pwd)
logdir=${CWD}/espnet/egs2/libritts/codec1/exp/codec_1/inference_model_tagespnet/libritts_encodec_16k_quantize_modelTrue_espnet_libritts_encodec_16k/test-clean/
scoredir=${CWD}

python3 -m src.bin.aggregate_results --logdir ${logdir} --scoredir ${scoredir}