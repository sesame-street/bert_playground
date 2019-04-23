#!/usr/bin/env bash


set -u
set -e
set -o pipefail

INCONFIG=$1
OUTCONFIG=$2

eval $(srs-config -config ${INCONFIG} -config default.config -dumpsh "cfg_")

local_output_dir="SRS-GO/data/scratch_local/triviaqa_dir"

if ! [ -d ${local_output_dir} ]; then
  mkdir ${local_output_dir}
else
  rm -f ${local_output_dir}/model.ckpt*
  rm -f ${local_output_dir}/checkpoint
fi

CUDA_VISIBLE_DEVICES=${cfg_cuda_visible_devices} python ${cfg_bert_repo_dir}/run_doc_qa.py \
  --vocab_file=${cfg_bert_model_dir}/vocab.txt \
  --bert_config_file=${cfg_bert_model_dir}/bert_config.json \
  --init_checkpoint=${cfg_squad_model_dir} \
  --do_train=True \
  --label_cleaning=${cfg_label_cleaning} \
  --train_file=${cfg_train_data} \
  --do_predict=False \
  --train_batch_size=${cfg_batch_size} \
  --num_train_epochs=${cfg_num_epoch} \
  --max_seq_length=${cfg_max_seq_length} \
  --doc_stride=${cfg_doc_stride} \
  --learning_rate=${cfg_learning_rate} \
  --use_tpu=False \
  --train_first_answer=${cfg_train_first_answer} \
  --max_num_doc_features=${cfg_max_num_doc_features} \
  --max_short_answers=${cfg_max_short_answers} \
  --max_num_answer_strings=${cfg_max_num_answer_strings} \
  --marginalize=${cfg_marginalize} \
  --posterior_distillation=${cfg_posterior_distillation} \
  --local_obj_alpha=${cfg_local_obj_alpha} \
  --pd_loss=${cfg_pd_loss} \
  --debug=${cfg_debug} \
  --version_2_with_negative=${cfg_has_negative} \
  --output_dir="${local_output_dir}/"

echo "local_output_dir ${local_output_dir}" > ${OUTCONFIG}
echo INCLUDE ${INCONFIG} >> ${OUTCONFIG}
