#!/usr/bin/env bash

news="$1"
cats="$2"
obase="$3"
template="$4"

cd /global/cscratch1/sd/dgold/ztfcoadd/job_scripts

python /pipeline/bin/makecoadd.py --input-frames ${news} --input-catalogs ${cats} \
               --output-basename=${obase} --nothreads

if $?; then

python /pipeline/bin/makesub.py --science-frames ${obase}.fits \
               --templates ${template}

fi
