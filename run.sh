#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# [[Model Config]]
model=encodec
# model=dac
# model=soundstream
model_tag=espnet/libritts_${model}_16k
quantize_model=True
quantize_modules=all
quantize_dtype=qint8

# [[Set recipe directory]]
CWD="$(pwd)"
espnet_root=/espnet
recipe_dir=${CWD}${espnet_root}/egs2/libritts/codec1

python=python3

feats_type=raw
data_feats=data
test_sets=test-clean
inference_nj=1
eval_nj=1
audio_format=flac
fs=16000

expdir=${CWD}/exp
tag=1
codec_exp="${expdir}/codec_${tag}"
inference_tag=""
inference_config=""
inference_args="--model_tag "${model_tag}" --quantize_model ${quantize_model} --quantize_modules ${quantize_modules} --quantize_dtype ${quantize_dtype}"
inference_model=${model_tag}
gpu_inference=false

# scoring_config=conf/score.yaml
scoring_config=${CWD}/conf/score.yaml
scoring_args=""
scoring_tag=""

. ./scripts/pipeline.sh