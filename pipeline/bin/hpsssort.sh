#!/usr/bin/env bash

#!/bin/bash


if [ $# -lt 1 ]; then
    echo usage is "hsi_file_sorter.script <name_of_file_containing_HPSS_file_names>"
    exit
fi

list=$1

OUTDIR=$HOME
if [ ! -z "$SCRATCH" ]; then
    OUTDIR=$SCRATCH
fi
temp_dir=$OUTDIR/hsi_sort_$RANDOM
#check that directory doesn't exist
while [ -d $temp_dir ]; do
    temp_dir=$OUTDIR/hsi_sort_$RANDOM
done

mkdir $temp_dir
tempin=$temp_dir/tempin_$RANDOM.txt

cat $list | awk '{print "ls -P",$0}' > $tempin


hsi -q "in "$tempin 2>&1 | grep -E ^FILE | awk '$3 != 0 {printf("%s %s %s %s\n",substr($6,0,6),substr($5\
,0,index($5,"+")-1),substr($5,index($5,"+")+1),$2)}' > $temp_dir/hsi_get.txt

#cat $temp_dir/hsi_get.txt | sort -n -k 1,2 -k 2,3 -k 3,4 | awk '{print $4}'
cat $temp_dir/hsi_get.txt | sort -k1,1 -k2,2n -k3,3n | awk '{print $0}'

#clean up
rm $temp_dir/hsi_get.txt
rm $tempin
rmdir $temp_dir



