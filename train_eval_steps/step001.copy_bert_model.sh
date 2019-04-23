#!/usr/bin/env bash

set -u
set -e
set -o pipefail

INCONFIG=$1
OUTCONFIG=$2

eval $(srs-config -config ${INCONFIG} -config default.config -dumpsh "cfg_")

local_bert_dir="SRS-GO/data/scratch_local/bert_dir"

if [ -d ${local_bert_dir} ]; then
  echo "Removing the existing local_bert_dir"
  rm -rf ${local_bert_dir}
fi

echo "Copies the bert repo"
cp -r ${cfg_bert_dir} ${local_bert_dir}

echo "Copies the bert model"
cp ${cfg_bert_model_path} "SRS-GO/data/scratch_local/"


bert_model_dir="SRS-GO/data/scratch_local/${cfg_bert_basename}"

if [ -d ${bert_model_dir} ]; then
  echo "Removing the existing bert_model_dir"
  rm -rf ${bert_model_dir}
fi

echo "Unzipping the bert model"
unzip "SRS-GO/data/scratch_local/${cfg_bert_basename}.zip" -d "SRS-GO/data/scratch_local"

squad_model_dir="SRS-GO/data/scratch_local/squad_dir"
tar -xzvf ${cfg_squad_model_tgz} -C "SRS-GO/data/scratch_local"

echo "bert_repo_dir ${local_bert_dir}" > ${OUTCONFIG}
echo "bert_model_dir ${bert_model_dir}" >> ${OUTCONFIG}
echo "squad_model_dir ${squad_model_dir}" >> ${OUTCONFIG}
echo INCLUDE ${INCONFIG} >> ${OUTCONFIG}
