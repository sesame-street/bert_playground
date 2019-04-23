#!/usr/bin/env bash


set -u
set -e
set -o pipefail

INCONFIG=$1
OUTCONFIG=$2

eval $(srs-config -config ${INCONFIG} -config default.config -dumpsh "cfg_")

eval_prediction_filename="SRS-GO/data/scratch_local/eval_prediction.json"

python extract_answers.py \
  --nbest_file=${cfg_prediction_dir}/nbest_predictions.json \
  --predictions_file=${eval_prediction_filename}

echo "eval_prediction_filename ${eval_prediction_filename}" > ${OUTCONFIG}
echo INCLUDE ${INCONFIG} >> ${OUTCONFIG}
