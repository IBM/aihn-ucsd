import os
import torch
from datetime import datetime

timestamp = datetime.now().strftime('%m%d_%H%M%S')

"""
Experiment settings
"""

# BAG AGGREGATION
use_bag_attn = False  # True = use attention on sents bag, False = avg of sents

# REL EMBEDDINGS EXPERIMENTS
# A: [CLS]
# B: entity mention pool
# C: [CLS] + entity mention pool
# D: e1_start + e2_start
# E: [CLS] + e1_start + e2_start
# F: e1_end + e2_end
# G: [CLS] + e1_end + e2_end
# H: e1_start + e1_end + e2_start + e2_end
# I: [CLS] + e1_start + e1_end + e2_start + e2_end
# J: middle mention pool
# K: [CLS] + middle mention pool
# L: e1_end + middle mention pool + e2_end
# M: [CLS] + e1_end + middle mention pool + e2_end
# N: e1_start + e1_end + middle mention pool + e2_start + e2_end
# O: [CLS] + e1_start + e1_end + middle mention pool + e2_start + e2_end
# P: entire sequence avg
# Q: [CLS] + entire sequence avg

rel_embedding = 'L'

# Bag sentence attention or bag sentence average
bag_representations = "attn" if use_bag_attn else "avg"

"""
Directories
"""
# MEDLINE Data Directories
DATA_DIR = "data"
MEDLINE_DIR = os.path.join(DATA_DIR, "MEDLINE")
medline_file = os.path.join(MEDLINE_DIR, "raw", "medline_abs.txt")

medline_unique_sents_file = os.path.join(MEDLINE_DIR, "processed", "medline_unique_sents.txt")
medline_spacy_sents = os.path.join(MEDLINE_DIR, "processed", "medline_spacy_sents.txt")
medline_linked_sents_file = os.path.join(MEDLINE_DIR, "processed", "umls.linked_sentences.jsonl")
groups_linked_sents_file = os.path.join(MEDLINE_DIR, "processed", "umls.linked_sentences_to_groups.jsonl")
groups_linked_sents_file_types = os.path.join(MEDLINE_DIR, "processed", "umls.linked_sentences_to_groups_types.jsonl")
reltext_all_combos = os.path.join(MEDLINE_DIR, "processed", "umls.reltxt_all_combos.pkl")

# UMLS Data Directories
UMLS_DIR = os.path.join(DATA_DIR, "UMLS")
mrrel_file = os.path.join(UMLS_DIR, "raw", "MRREL.RRF")
mrconso_file = os.path.join(UMLS_DIR, "raw", "MRCONSO.RRF")
mrsty_file = os.path.join(UMLS_DIR, "raw", "MRSTY.RRF")

umls_txt_to_cui = os.path.join(UMLS_DIR, "processed", "umls.txt_to_cui.pkl")
umls_cui_to_txts = os.path.join(UMLS_DIR, "processed", "umls.cui_to_txts.pkl")
umls_reltxt_to_groups = os.path.join(UMLS_DIR, "processed", "umls.reltxt_to_groups.pkl")
umls_cui_to_types = os.path.join(UMLS_DIR, "processed", "umls.cui_to_types.pkl")
umls_text_to_type = os.path.join(UMLS_DIR, "processed", "umls.text_to_type.pkl")

# Make all UMLS and MEDLINE directories
make_dirs = ["raw", "processed"]
for directory in make_dirs:
    os.makedirs(os.path.join(MEDLINE_DIR, directory), exist_ok=True)
    os.makedirs(os.path.join(UMLS_DIR, directory), exist_ok=True)

# Processed Data Directories
SAVE_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(SAVE_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Processed triples, relations, and entity name files
entities_file = os.path.join(SAVE_DIR, "entities.txt")
relations_file = os.path.join(SAVE_DIR, "relations.txt")
triples_file_all = os.path.join(SAVE_DIR, "triples_all.tsv")
triples_file_train = os.path.join(SAVE_DIR, "triples_train.tsv")
triples_file_dev = os.path.join(SAVE_DIR, "triples_dev.tsv")
triples_file_test = os.path.join(SAVE_DIR, "triples_test.tsv")

entities_file_types = os.path.join(SAVE_DIR, "entities_types.txt")
relations_file_types = os.path.join(SAVE_DIR, "relations_types.txt")
triples_types_file_all = os.path.join(SAVE_DIR, "triples_types_all.tsv")
triples_types_file_train = os.path.join(SAVE_DIR, "triples_types_train.tsv")
triples_types_file_dev = os.path.join(SAVE_DIR, "triples_types_dev.tsv")
triples_types_file_test = os.path.join(SAVE_DIR, "triples_types_test.tsv")

# Complete data files, splits
complete_train = os.path.join(SAVE_DIR, "complete_train.txt")
complete_dev = os.path.join(SAVE_DIR, "complete_dev.txt")
complete_test = os.path.join(SAVE_DIR, "complete_test.txt")
lower_half_trips = os.path.join(SAVE_DIR, "lower_80.pkl")
lower_half_trips_b = os.path.join(SAVE_DIR, "lower_80_b.pkl")

# Types
complete_types_all = os.path.join(SAVE_DIR, "complete_types_all.txt")
complete_types_train = os.path.join(SAVE_DIR, "complete_types_train.txt")
complete_types_dev = os.path.join(SAVE_DIR, "complete_types_dev.txt")
complete_types_test = os.path.join(SAVE_DIR, "complete_types_test.txt")

# Features files
feats_file_train = os.path.join(FEATURES_DIR, "train.pt")
feats_file_dev = os.path.join(FEATURES_DIR, "dev.pt")
feats_file_test = os.path.join(FEATURES_DIR, "test.pt")

feats_file_types_train = os.path.join(FEATURES_DIR, "types_train.pt")
feats_file_types_dev = os.path.join(FEATURES_DIR, "types_dev.pt")
feats_file_types_test = os.path.join(FEATURES_DIR, "types_test.pt")

# Model output directory
model_config = [
    timestamp + '.',
    bag_representations + '.',
    rel_embedding + '.',
]
model_config = "".join(model_config)
output_dir = os.path.join("saved_models", model_config)
test_ckpt = output_dir

"""
Experiment setup
"""
SEED = 2019

# Entity linking options
min_sent_char_len_linker = 32
max_sent_char_len_linker = 256

# Relationship option
min_rel_group = 10
max_rel_group = 1500

# Bag options
bag_size = 16

# BERT model settings
pretrained_model_dir = "biobert"
do_lower_case = False
max_seq_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_batch_size = 2
eval_batch_size = 1
gradient_accumulation_steps = 1
num_train_epochs = 300
learning_rate = 2e-5
adam_epsilon = 1e-8
warmup_percent = 0.01
max_grad_norm = 1.0
weight_decay = 0.
logging_steps = 5000
save_steps = 5000
max_steps = 150000
early_stop = 20