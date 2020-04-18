#!/usr/bin/env bash

export news="$1"
export cats="$2"
export coadd="$3"
export template="$4"
export subid="$5"
export coaddid="$6"
export sub="$7"

python /pipeline/bin/movedeps.py $news $coadd

if [ $? -eq 0 ]; then 

echo "python /pipeline/bin/makecoadd.py --input-frames $news --input-catalogs $cats --outfile-path=$coadd && \
python /pipeline/bin/log_image.py $coadd $coaddid Stack"

python /pipeline/bin/makecoadd.py --input-frames $news --input-catalogs $cats --outfile-path=$coadd && \
python /pipeline/bin/log_image.py $coadd $coaddid Stack

if [ $? -eq 0 ]; then

echo "python /pipeline/bin/makesub.py --science-frames $coadd --templates $template && \
python /pipeline/bin/log_image.py $sub $subid MultiEpochSubtraction"

python /pipeline/bin/makesub.py --science-frames $coadd --templates $template && \
python /pipeline/bin/log_image.py $sub $subid MultiEpochSubtraction

fi

fi 
