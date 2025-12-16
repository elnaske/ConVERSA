#!/usr/bin/env bash

set -e
set -u
set -o pipefail

source espnet/tools/venv/bin/activate

_dir=exp/codec_2/encodec/True/qint8/

python3 -m src.bin.aggregate_results --logdir ${_dir} --scoredir ${_dir}

deactivate