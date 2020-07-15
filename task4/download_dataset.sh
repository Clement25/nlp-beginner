#!/bin/bash

train=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/train.txt
valid=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/valid.txt
test=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/test.txt

if [ ! -n "$1" ]; then
    folder=conll2003
fi

mkdir $folder | wget --show-progress $train && mv train.txt $folder
wget --show-progress $valid && mv valid.txt $folder
wget --show-progress $test && mv test.txt $folder