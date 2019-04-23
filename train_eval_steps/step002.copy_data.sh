#!/usr/bin/env bash

set -u
set -e
set -o pipefail

INCONFIG=$1
OUTCONFIG=$2

eval $(srs-config -config ${INCONFIG} -config default.config -dumpsh "cfg_")

data_dir="SRS-GO/data/scratch_local/local_data_dir"

if ! [ -d ${data_dir} ]; then
  mkdir ${data_dir}
fi

cp ${cfg_data_dir}/${cfg_train_filename} ${data_dir}/
cp ${cfg_data_dir}/${cfg_dev_filename} ${data_dir}/

echo "train_data ${data_dir}/${cfg_train_filename}" > ${OUTCONFIG}
echo "dev_data ${data_dir}/${cfg_dev_filename}" >> ${OUTCONFIG}
echo INCLUDE ${INCONFIG} >> ${OUTCONFIG}

