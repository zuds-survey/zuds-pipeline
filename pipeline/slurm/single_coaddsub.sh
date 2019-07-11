#!/usr/bin/env bash

news="$1"
cats="$2"
coadd="$3"
template="$4"
subid="$5"
coaddid="$6"
sub="$7"

python /pipeline/bin/makecoadd.py --input-frames $news --input-catalogs $cats --outfile-path=$coadd && \
python /pipeline/bin/log_image.py $coadd $coaddid Stack

if [ $? -eq 0 ]; then

python /pipeline/bin/makesub.py --science-frames $coadd --templates $template && \
python /pipeline/bin/log_image.py $sub $subid MultiEpochSubtraction

fi
