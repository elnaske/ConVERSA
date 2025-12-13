#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
    
quantize_modules="Conv1d LSTM Linear"

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
audio_format=wav
fs=16000

# [[Model Config]]
for model in dac_ encodec_ soundstream; do
    # model=encodec
    # model=dac
    # model=soundstream
    model_tag=espnet/libritts_${model}16k

    cnf=config_2

    expdir=${CWD}/exp
    tag=${cnf}
    codec_exp="${expdir}/codec_${tag}"
    inference_tag=""
    inference_config=""
    
    inference_model=${model_tag}
    gpu_inference=false
    gpu_eval=false

    # scoring_config=conf/score.yaml
    scoring_config=${CWD}/conf/score.yaml
    scoring_args=""
    scoring_tag=""


    # run three times for levels of precision
    benchmark_config=${CWD}/conf/${cnf}.yaml

    quantize_model=true
    quantize_dtype=qint8

    inference_args="--model_tag "${model_tag}" --quantize_model ${quantize_model} --quantize_modules ${quantize_modules} --quantize_dtype ${quantize_dtype}"

    echo int8 quantization
    . ./scripts/pipeline.sh

    quantize_model=true
    quantize_dtype=float16

    inference_args="--model_tag "${model_tag}" --quantize_model ${quantize_model} --quantize_modules ${quantize_modules} --quantize_dtype ${quantize_dtype}"

    echo float16 quantization
    cd ${CWD}
    . ./scripts/pipeline.sh

    quantize_model=false

    inference_args="--model_tag "${model_tag}" --quantize_model ${quantize_model} --quantize_modules ${quantize_modules} --quantize_dtype ${quantize_dtype}"

    echo no_quanzitation
    cd ${CWD}
    . ./scripts/pipeline.sh

    cd ${CWD}
done