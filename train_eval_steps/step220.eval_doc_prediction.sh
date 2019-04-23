#!/usr/bin/env bash


set -u
set -e
set -o pipefail

INCONFIG=$1
OUTCONFIG=$2

eval $(srs-config -config ${INCONFIG} -config default.config -dumpsh "cfg_")

python triviaqa_evaluation.py \
  --dataset_file ${cfg_eval_data} \
  --prediction_file ${cfg_eval_prediction_filename} \
  --out_file ${cfg_prediction_dir}/eval.metrics

cp ${cfg_prediction_dir}/eval.metrics SRS-GO/data/

echo INCLUDE ${INCONFIG} >> ${OUTCONFIG}
