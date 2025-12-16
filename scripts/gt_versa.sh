#!/usr/bin/env bash
# Script for VERSA scoring on grount-truth audio

set -e
set -u
set -o pipefail

source espnet/tools/venv/bin/activate

CWD=$(pwd)
_data=${CWD}/espnet/egs2/libritts/codec1/data/test-clean
_gt_wavscp="${_data}/wav.scp"
_gt_text="${_data}/text"
scoring_config=${CWD}/conf/score.yaml

cd espnet/egs2/libritts/codec1

# python3 -m versa.bin.scorer \
#     --pred "${_gt_wavscp}" \
#     --gt "${_gt_wavscp}" \
#     --text "${_gt_text}" \
#     --output_file "${CWD}/result.1.txt" \
#     --score_config "${scoring_config}"

python3 pyscripts/utils/aggregate_eval.py \
    --logdir "${CWD}" \
    --scoredir "${CWD}" \
    --nj 1

deactivate