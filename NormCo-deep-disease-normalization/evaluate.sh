#!/bin/bash

data_dir=$1
model_file=$2

. setenv.sh

python entity_normalization/eval.py \
    --weight_init $model_file \
    --disease_dict $data_dir/concept_dict.tsv \
    --vocab_file $data_dir/vocab.txt \
    --embeddings_file $data_dir/word_embeddings_init.npy \
    --labels_file $data_dir/tagger_labels.txt \
    --labels_with_abbs_file $data_dir/tagger_labels.txt_with_abbreviations \
    --banner_tags $data_dir/tagger_data.txt \
    --hierarchy_file $data_dir/hierarchy.tsv \
    --model GRU \
    --sequence_len 20 \
    --scoring_type euclidean \
    --output_dim 200 \
    --mention_only
