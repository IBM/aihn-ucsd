import json
import os
import pickle
import random

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

import config
from utils.utils import TriplesReader as read_triples
from utils.utils import read_relations, read_entities


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.

    Taken from:
    	https://github.com/thunlp/OpenNRE/blob/master/opennre/framework/utils.py

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def set_seed():
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_metrics(logits, labels, groups, set_type, logger, ent_types=False):
    # Read relation mappings and triples
    if ent_types:
        rel2idx = read_relations(config.relations_file_types)
        entity2idx = read_entities(config.entities_file_types)
        if set_type == "dev":
            triples_file = config.triples_types_file_dev
        else:
            triples_file = config.triples_file_test
            entity2idx = read_entities(config.entities_file)
    else:
        rel2idx = read_relations(config.relations_file)
        entity2idx = read_entities(config.entities_file)
        if set_type == "dev":
            triples_file = config.triples_file_dev
        else:
            triples_file = config.triples_file_test

    # Read triples
    triples = set()
    print('Loaded ', triples_file)

    for src, rel, tgt in read_triples(triples_file):
        if rel != "na":
            triples.add((entity2idx[src], rel, entity2idx[tgt]))

    # RE predictions
    probas = torch.nn.Softmax(-1)(logits).squeeze()
    re_preds = list()

    for i in range(probas.size(0)):
        group = groups[i]
        src, tgt = group[0].item(), group[1].item()
        for rel, rel_idx in rel2idx.items():
            if rel != "na":
                score = probas[i][rel_idx].item()
                re_preds.append({
                    "src": src, "tgt": tgt,
                    "relation": rel,
                    "score": score
                })


    # Adopted from:
    # https://github.com/thunlp/OpenNRE/blob/master/opennre/framework/data_loader.py#L230
    sorted_re_preds = sorted(re_preds, key=lambda x: x["score"], reverse=True)
    sorted_re_preds = non_dup_ordered_seq(sorted_re_preds)
    P = list()
    R = list()
    correct = 0
    total = len(triples)

    for i, item in enumerate(sorted_re_preds):
        relation = item["relation"]
        src, tgt = item["src"], item["tgt"]
        if (src, relation, tgt) in triples:
            correct += 1
        P.append(float(correct) / float(i + 1))
        R.append(float(correct) / float(total))

    auc = metrics.auc(x=R, y=P)
    P = np.array(P)
    R = np.array(R)

    f1 = (2 * P * R / (P + R + 1e-20)).max()

    # Added metrics
    added_metrics = {}
    for n in range(100, 1000, 100): # 100, 200, etc recall
        added_metrics['P@{}'.format(n)] = sum(P[:n]) / n

    for n in range(2000, total, 2000):
        added_metrics['P@{}'.format(n)] = sum(P[:n]) / n
    added_metrics['P@{}'.format(total)] = sum(P[:total]) / total

    # Accuracy
    na_idx = rel2idx["na"]

    preds = torch.argmax(torch.nn.Softmax(-1)(logits), -1)
    acc = float((preds == labels).long().sum()) / labels.size(0)
    pos_total = (labels != na_idx).long().sum()
    pos_correct = ((preds == labels).long() * (labels != na_idx).long()).sum()
    if pos_total > 0:
        pos_acc = float(pos_correct) / float(pos_total)
    else:
        pos_acc = 0
    logger.info(" accuracy = %s", str(acc))
    logger.info(" pos_accuracy = %s", str(pos_acc))

    results = {"P": list(P[:5]), "R": list(R[:5]), "F1": f1, "AUC": auc, "accuracy: ": str(acc),
               "pos_accuracy: ": str(pos_acc)}
    results.update(added_metrics)

    return results


def save_eval_results(results, eval_dir, set_type, logger, prefix=""):
    os.makedirs(eval_dir, exist_ok=True)
    output_eval_file = os.path.join(eval_dir, "eval_results.txt")

    with open(output_eval_file, "w") as wf:
        logger.info("***** {} results {} *****".format(set_type, prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            wf.write("%s = %s\n" % (key, str(results[key])))


def load_dataset(set_type, logger, ent_types=True):
    if set_type == "train":
        if ent_types:
            features_file = config.feats_file_types_train
        else:
            features_file = config.feats_file_train
    elif set_type == "dev":
        if ent_types:
            features_file = config.feats_file_types_dev
        else:
            features_file = config.feats_file_dev
    else:
        if ent_types:
            features_file = config.feats_file_types_test
        else:
            features_file = config.feats_file_test

    logger.info("Loading features from cached file %s", features_file)
    features = torch.load(features_file)

    all_input_ids = torch.cat([f["input_ids"].unsqueeze(0) for f in features]).long()
    all_entity_ids = torch.cat([f["entity_ids"].unsqueeze(0) for f in features]).long()
    all_attention_mask = torch.cat([f["attention_mask"].unsqueeze(0) for f in features]).long()
    all_groups = torch.cat([torch.tensor(f["group"]).unsqueeze(0) for f in features]).long()
    all_labels = torch.tensor([f["label"] for f in features]).long()
    if ent_types:  # include ent names within ent types
        all_names = [f["ent_names"] for f in features]
        all_names = convert_names_to_cuis(all_names)
        dataset = TensorDataset(all_input_ids, all_entity_ids, all_attention_mask, all_groups, all_labels, all_names)
    else:
        dataset = TensorDataset(all_input_ids, all_entity_ids, all_attention_mask, all_groups, all_labels)
    return dataset


def convert_names_to_cuis(l_names):
    entity2idx = read_entities(config.entities_file)
    lc = []
    for l_bag in l_names:
        lb = []
        for l_group in l_bag:
            lb.append((entity2idx[l_group[0]], entity2idx[l_group[1]]))
        lc.append(lb)
    lc = torch.IntTensor(lc)
    return lc


# cf. https://stackoverflow.com/a/480227
def non_dup_ordered_seq(seq):
    seen = set()
    seen_add = seen.add
    non_dup_seq = list()
    for item in seq:
        relation = item["relation"]
        src, tgt = item["src"], item["tgt"]
        triple = (src, relation, tgt)
        if not (triple in seen or seen_add(triple)):
            non_dup_seq.append(item)
    return non_dup_seq


def evaluate(model, logger, set_type="dev", prefix=""):
    eval_output_dir = config.output_dir
    eval_dataset = load_dataset(set_type, logger, ent_types=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    eval_logits, eval_labels, eval_preds, eval_groups, eval_dirs, eval_names = [], [], [], [], [], []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "entity_ids": batch[1],
                "attention_mask": batch[2],
                "labels": batch[4],
                "is_train": False
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        # Evaluate original triples, not abstract triples
        trip_names = batch[5].detach().cpu().squeeze()
        unique_trips_in_group = set()
        for trip in trip_names:
            if trip not in unique_trips_in_group:
                unique_trips_in_group.add(trip)
                eval_labels.append(inputs["labels"].detach().cpu())
                eval_logits.append(logits.detach().cpu())
                eval_groups.append(batch[3].detach().cpu())  # groups
                eval_names.append(trip)  # names
                eval_preds.append(torch.argmax(logits.detach().cpu(), dim=1).item())

    del model, batch, logits, tmp_eval_loss, eval_dataloader, eval_dataset  # memory mgmt

    eval = {
        'loss': eval_loss / nb_eval_steps,
        'labels': torch.stack(eval_labels),  # B,
        'logits': torch.stack(eval_logits),  # B x C
        'preds': np.asarray(eval_preds),
        'groups': torch.stack(eval_groups),  # B x 2
        'names' : torch.stack(eval_names)
    }

    # Get all positive relationship lables
    rel2idx = read_relations(config.relations_file_types)
    pos_rel_idxs = list(rel2idx.values())
    rel_idx_na = rel2idx['na']
    del pos_rel_idxs[rel_idx_na]

    a = accuracy_score(eval['labels'].numpy(), eval['preds'])
    p, r, f1, support = precision_recall_fscore_support(eval['labels'].numpy(), eval['preds'], average='micro',
                                                        labels=pos_rel_idxs)
    logger.info('Accuracy (including "NA"): {}\nP: {}, R: {}, F1: {}'.format(a, p, r, f1))
    results = {}
    results['new_results'] = {
        'acc_with_na': a,
        'scikit_precision': p,
        'scikit_recall': r,
        'scikit_f1': f1,
        "loss": eval_loss,
        "counter": eval['labels'].shape
    }

    results['original'] = compute_metrics(eval['logits'], eval['labels'], eval['names'], set_type, logger,
                                          ent_types=True)
    logger.info("Results: %s", results)

    # Save evaluation results
    with open(os.path.join(config.output_dir, set_type + "_metrics.txt"), "w") as wf:
        json.dump(results, wf, indent=4)

    # Save evaluation raw data
    with open(os.path.join(config.output_dir, set_type + "_raw_eval_data.pkl"), "wb") as wf:
        pickle.dump(eval, wf)

    return results
