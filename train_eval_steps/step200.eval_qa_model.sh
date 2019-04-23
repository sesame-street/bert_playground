#!/usr/bin/env bash

set -u
set -e
set -o pipefail

INCONFIG=$1
OUTCONFIG=$2

eval $(srs-config -config ${INCONFIG} -config default.config -dumpsh "cfg_")

prediction_dir="SRS-GO/data/scratch_local/prediction_dir"

if ! [ -d ${prediction_dir} ]; then
  mkdir ${prediction_dir}
fi

CUDA_VISIBLE_DEVICES=${cfg_cuda_visible_devices} python ${cfg_bert_repo_dir}/run_doc_qa.py \
  --do_train=False \
  --do_predict=True \
  --predict_file=${cfg_dev_data} \
  --vocab_file=${cfg_bert_model_dir}/vocab.txt \
  --bert_config_file=${cfg_bert_model_dir}/bert_config.json \
  --init_checkpoint=${cfg_local_output_dir} \
  --max_seq_length=${cfg_max_seq_length} \
  --doc_stride=${cfg_doc_stride} \
  --use_tpu=False \
  --version_2_with_negative=${cfg_has_negative} \
  --doc_normalize=${cfg_doc_normalize} \
  --prob_trans_func=${cfg_prob_trans_func} \
  --output_dir="${prediction_dir}/"

echo "prediction_dir ${prediction_dir}" > ${OUTCONFIG}
echo INCLUDE ${INCONFIG} >> ${OUTCONFIG}
