#!/bin/bash

data_dir=$1
output_dir=$2

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

. setenv.sh

#With context
python entity_normalization/train.py \
    --disease_dict $data_dir/concept_dict.tsv \
    --train_data $data_dir/trainset_mentions_preprocessed.npz \
    --coherence_data $data_dir/trainset_coherence_preprocessed.npz \
    --dictionary_data $data_dir/trainset_dictionary_preprocessed.npz \
    --distant_data $data_dir/trainset_distant_preprocessed.npz \
    --vocab_file $data_dir/vocab.txt \
    --embeddings_file $data_dir/word_embeddings_init.npy \
    --save_file_name $output_dir/model.pth \
    --labels_file $data_dir/dev_data.txt \
    --disease_embeddings_file $data_dir/concept_embeddings_init.npy \
    --model GRU \
    --num_epochs 100 \
    --batch_size 10 \
    --sequence_len 20 \
    --num_neg 30 \
    --lr 0.0005 \
    --scoring_type euclidean \
    --optimizer adam \
    --loss maxmargin \
    --threads 10 \
    --save_every 0 \
    --eval_every 1 \
    --output_dim 200 \
