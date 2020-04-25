#!/bin/bash

path=$HOME/.local/share/jupyter/kernels/zuds:0.1dev
echo $path
mkdir -p $path
cp kernel.json $path
sed -i "s|HOMEDIR|$HOME|g" $path/kernel.json

