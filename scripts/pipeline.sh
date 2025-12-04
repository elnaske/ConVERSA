#!/usr/bin/env bash
# Adapted from codec.sh  in espnet/egs2/libritts/codec1
# Downloads and prepares test-clean subset of libritts

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

# Configs, will go in another file later
CWD="$(pwd)"
espnet_root=/espnet
recipe_dir=${CWD}${espnet_root}/egs2/libritts/codec1
cd ${recipe_dir}

model_tag=espnet/libritts_encodec_16k
python=python3

feats_type=raw
data_feats=dump/raw
test_sets=test-clean
nj=32
inference_nj=32
audio_format=flac
fs=16000

expdir=exp
tag=1
codec_exp="${expdir}/codec_${tag}"
inference_tag=""
inference_config=""
inference_args="--model_tag "${model_tag}" --quantize_model True"
inference_model=${model_tag}
gpu_inference=false

scoring_config=conf/score.yaml
scoring_args=""
scoring_tag=""

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

if [ -z "${scoring_tag}" ]; then
    if [ -n "${scoring_config}" ]; then
        scoring_tag="$(basename "${scoring_config}" .yaml)"
    else
        scoring_tag=scoring
    fi
    # Add overwritten arg's info
    if [ -n "${scoring_args}" ]; then
        scoring_tag+="$(echo "${scoring_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

if ${gpu_inference}; then
    _cmd="${cuda_cmd}"
    _ngpu=1
else
    _cmd="${decode_cmd}"
    _ngpu=0
fi

_opts=
if [ -n "${inference_config}" ]; then
    _opts+="--config ${inference_config} "
fi

_scp=wav.scp
if [[ "${audio_format}" == *ark* ]]; then
    _type=kaldi_ark
else
    # "sound" supports "wav", "flac", etc.
    _type=sound
fi

mkdir -p "${codec_exp}/${inference_tag}"

for dset in ${test_sets}; do
    _data="${data_feats}/${dset}"
    _dir="${codec_exp}/${inference_tag}/${dset}"
    _logdir="${_dir}/log"
    mkdir -p "${_logdir}"

    # 0. Copy feats_type
    cp "${_data}/feats_type" "${_dir}/feats_type"

    # 1. Split the key file
    key_file=${_data}/wav.scp
    split_scps=""
    _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
    echo Number of jobs: ${_nj}
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    log "Collecting general model info"
    PYTHONPATH=${CWD} ${python} -m src.bin.model_info \
        --ngpu "${_ngpu}" \
        --output_dir "${_logdir}" \
        ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/model_info.log) ; exit 1; }

    log "Decoding started... log: '${_logdir}/codec_inference.*.log'"
    # shellcheck disable=SC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/codec_inference.JOB.log \
        PYTHONPATH=${CWD} ${python} -m src.bin.inference \
            --ngpu "${_ngpu}" \
            --data_path_and_name_and_type ${recipe_dir}/${_data}/${_scp},audio,${_type} \
            --key_file "${_logdir}"/keys.JOB.scp \
            --output_dir "${_logdir}"/output.JOB \
            ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/codec_inference.*.log) ; exit 1; }

    if [ -e "${_logdir}/output.${_nj}/codes" ]; then
        mkdir -p "${_dir}"/codes
        for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/codes/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/codes/feats.scp"
    fi
    if [ -e "${_logdir}/output.${_nj}/wav" ]; then
        mkdir -p "${_dir}"/wav
        for i in $(seq "${_nj}"); do
            mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
            rm -rf "${_logdir}/output.${i}"/wav
        done
        find "${_dir}/wav" -name "*.wav" | while read -r line; do
            echo "$(basename "${line}" .wav) ${line}"
        done | LC_ALL=C sort -k1 > "${_dir}/wav/wav.scp"
    fi
done

for dset in ${test_sets}; do
    _data="${data_feats}/${dset}"
    _gt_wavscp="${_data}/wav.scp"
    _dir="${codec_exp}/${inference_tag}/${dset}"
    _gen_wavscp="${_dir}/wav/wav.scp"

    log "Begin evaluation on ${dset}, results are written under ${_dir}"

    # 1. Split the key file
    _scoredir="${_dir}/${scoring_tag}"
    _logdir="${_scoredir}/log"
    mkdir -p ${_scoredir}
    mkdir -p ${_logdir}

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${inference_nj}" "$(<${_gen_wavscp} wc -l)" )
    _nj=1

    key_file=${_gen_wavscp}
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/test.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 3. Submit jobs
    log "Evaluation started... log: '${_logdir}/codec_evaluate.*.log'"
    # shellcheck disable=SC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/codec_evaluate.JOB.log \
        ${python} -m versa.bin.scorer \
            --pred "${_logdir}"/test.JOB.scp \
            --gt "${_gt_wavscp}" \
            --output_file "${_logdir}/result.JOB.txt" \
            --score_config "${scoring_config}" \
            ${scoring_args} || { cat $(grep -l -i error "${_logdir}"/codec_evaluate.*.log) ; exit 1; }

    # 4. Aggregate the VERSA results
    ${python} pyscripts/utils/aggregate_eval.py \
        --logdir "${_logdir}" \
        --scoredir "${_scoredir}" \
        --nj "${_nj}"

done

# Aggregate results from all steps
_logdir=${CWD}/espnet/egs2/libritts/codec1/exp/codec_1/inference_model_tagespnet/libritts_encodec_16k_quantize_modelTrue_espnet_libritts_encodec_16k/test-clean/
_scoredir=${CWD}

cd ${CWD}

python3 -m src.bin.aggregate_results --logdir ${_logdir} --scoredir ${_scoredir}

# 5. Show results
echo "Result saved at ${_scoredir}/results.json"
