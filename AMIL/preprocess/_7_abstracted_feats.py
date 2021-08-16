import logging

import torch
from transformers import BertTokenizer

import config
from utils.utils import JsonlReader, read_entities, read_relations

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_jsonl(jsonl, tokenizer, entity2idx, relation2idx, idx, max_seq_length=128,
                   e1_tok="$", e2_tok="^"):
    # Group
    src, tgt = jsonl["group"]
    relation = jsonl["relation"].lower()
    ent_names = jsonl["ent_names"]
    input_ids = list()
    entity_ids = list()
    attention_mask = list()

    for sent in jsonl["sentences"]:
        encoded = tokenizer.encode_plus(sent, max_length=max_seq_length, pad_to_max_length=True, return_tensors='pt')
        input_ids_i = encoded["input_ids"]
        attention_mask_i = encoded["attention_mask"]
        entity_ids_i = torch.zeros(max_seq_length)

        # Can happen for long sentences when entity markers go out of boundary, ignore such cases
        e1_start, e1_end = torch.nonzero(input_ids_i[0] == tokenizer.vocab[e1_tok]).flatten()

        entity_ids_i[e1_start + 1:e1_end] = 1

        e2_start, e2_end = torch.nonzero(input_ids_i[0] == tokenizer.vocab[e2_tok]).flatten()

        entity_ids_i[e2_start + 1:e2_end] = 2
        input_ids.append(input_ids_i)
        entity_ids.append(entity_ids_i.unsqueeze(0))
        attention_mask.append(attention_mask_i)

    # Happened once -- somehow?!
    if len(input_ids) == 0:
        return []

    group = (entity2idx[src], entity2idx[tgt])

    features = [dict(
        input_ids=torch.cat(input_ids),
        entity_ids=torch.cat(entity_ids),
        attention_mask=torch.cat(attention_mask),
        label=relation2idx[relation],
        group=group,
        ent_names=ent_names
    ), ]

    return features


def load_tokenizer(model_dir, do_lower_case=False):
    return BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)


def create_features(jsonl_fname, tokenizer, output_fname, entity2idx, relation2idx,
                    max_seq_length=128, e1_tok="$", e2_tok="^"):
    jr = list(iter(JsonlReader(jsonl_fname)))
    features = list(), list()
    logger.info("Loading {} lines from complete data txt file.".format(len(jr)))
    for idx, jsonl in enumerate(jr):
        if idx % 10000 == 0 and idx != 0:
            logger.info("Created {} features".format(idx))

        feats = tokenize_jsonl(
            jsonl, tokenizer, entity2idx, relation2idx, idx,
            max_seq_length, e1_tok, e2_tok
        )
        features.extend(feats)
    torch.save(features, output_fname)
    logger.info("Saved {} lines of features.".format(len(features)))


if __name__ == "__main__":
    tokenizer = load_tokenizer(config.pretrained_model_dir, config.do_lower_case)
    entity2idx = read_entities(config.entities_file_types)
    relation2idx = read_relations(config.relations_file_types)
    logger.info("Read {} unique entities and {} unique relations.".format(len(entity2idx), len(relation2idx)))

    files = [
        (config.complete_types_train, config.feats_file_types_train),
        (config.complete_types_dev, config.feats_file_types_dev),
        (config.complete_types_test, config.feats_file_types_test)
    ]

    for input_fname, output_fname in files:
        logger.info("Creating features for input `{}` ...".format(input_fname))
        create_features(
            input_fname, tokenizer, output_fname, entity2idx,
            relation2idx, config.max_seq_length
        )
        logger.info("Saved features at `{}` ...".format(output_fname))
