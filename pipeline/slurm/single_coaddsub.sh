#!/usr/bin/env bash

news="$1"
cats="$2"
coadd="$3"
template="$4"

python /pipeline/bin/makecoadd.py --input-frames $news --input-catalogs $cats --outfile-path=$coadd

if [ $? -eq 0 ]; then

python /pipeline/bin/makesub.py --science-frames $coadd --templates $template

fi
