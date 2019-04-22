#!/bin/bash

data_dir=$1
concept_type=$2
preprocessed_data_dir=$3

traindev_file=$data_dir/traindev_set.PubTator.txt
test_file=$data_dir/test_set.PubTator.txt
train_pmids=$data_dir/train_pmids.txt
dev_pmids=$data_dir/dev_pmids.txt
test_pmids=$data_dir/test_pmids.txt
concept_dict=$data_dir/concept_dict.tsv
tagger_entities=$data_dir/tagger_entities.txt
tagger_labels=$data_dir/tagger_labels.txt
pretrained_embeddings=./data/embeddings/PubMed-and-PMC-w2v.bin
abbreviations_file=$data_dir/abbreviations.tsv
distant_supervision=$data_dir/distant_supervision.txt

if [ ! -d $preprocessed_data_dir ]; then
    mkdir -p $preprocessed_data_dir
fi

. setenv.sh

python data/generate_data.py \
    --traindev_file $traindev_file \
    --test_file $test_file \
    --train_pmids $train_pmids \
    --dev_pmids $dev_pmids \
    --test_pmids $test_pmids \
    --concept_dict $concept_dict \
    --tagger_entities_file $tagger_entities \
    --use_unk_concept \
    --tagger_labels_file $tagger_labels \
    --pretrained_embeddings $pretrained_embeddings \
    --abbreviations_file $abbreviations_file \
    --distant_supervision_data $distant_supervision \
    --vocab_file $preprocessed_data_dir/vocab.txt \
    --word_embeddings_init_file $preprocessed_data_dir/word_embeddings_init.npy \
    --concept_embeddings_init_file $preprocessed_data_dir/concept_embeddings_init.npy \
    --trainset_mentions_preprocessed_file $preprocessed_data_dir/trainset_mentions_preprocessed.npz \
    --trainset_dictionary_preprocessed_file $preprocessed_data_dir/trainset_dictionary_preprocessed.npz \
    --trainset_coherence_preprocessed_file $preprocessed_data_dir/trainset_coherence_preprocessed.npz \
    --trainset_distant_preprocessed_file $preprocessed_data_dir/trainset_distant_preprocessed.npz \
    --test_data_file $preprocessed_data_dir/test_data.txt \
    --dev_data_file $preprocessed_data_dir/dev_data.txt \
    --tagger_data_file $preprocessed_data_dir/tagger_data.txt \
    --tagger_labels_output $preprocessed_data_dir/tagger_labels.txt 


cp $concept_dict $preprocessed_data_dir/concept_dict.tsv
cp $data_dir/hierarchy.tsv $preprocessed_data_dir
cp $data_dir/preprocessed_synthetic_data.npz $preprocessed_data_dir
