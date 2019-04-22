#!/bin/bash

data_dir=$1
model_file=$2
tags_file=$3

. setenv.sh

python entity_normalization/normalize.py \
    --weight_init $model_file \
    --disease_dict $data_dir/concept_dict.tsv \
    --vocab_file $data_dir/vocab.txt \
    --embeddings_file $data_dir/word_embeddings_init.npy \
    --tags_file $tags_file \
    --model GRU \
    --sequence_len 20 \
    --scoring_type euclidean \
    --output_dim 200 \
