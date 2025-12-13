#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# [[Model Config]]
# model=encodec_
model=dac_
# model=soundstream
model_tag=espnet/libritts_${model}16k
quantize_model=true
quantize_modules="all"
quantize_dtype=qint8

# [[Input directories]]
CWD="$(pwd)"
recipe_dir=${CWD}/espnet/egs2/libritts/codec1
data_feats=data
test_sets=test-clean

# [[Output directory]]
expdir=${CWD}/exp
tag=1000

# [[Config files]]
benchmark_config=${CWD}/conf/config.yaml
scoring_config=${CWD}/conf/score.yaml

# [[Jobs]]
inference_nj=1
eval_nj=1

. ./scripts/pipeline.sh