"""

Script adopted from:
    https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

"""

import collections
import logging

from transformers import BertConfig

from model.model import BertForDistantRE
from utils.train_utils import *
from utils.utils import idx_to_rel, idx_to_ent, trip_set

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def long_tail_split(all, upper_20=False):
    with open(config.lower_half_trips, "rb") as wf:
        lower_80_ids = pickle.load(wf)

    lower_80_results = collections.defaultdict(list)
    kept, total = 0, 0
    index_to_rel_types = idx_to_rel(config.relations_file_types)
    rel_to_idx_names = read_relations(config.relations_file)
    logger.info('SIZE OF LOWER 80 SET: {}'.format(len(lower_80_ids)))
    logger.info('SIZE OF ALL DICT: {}'.format(len(all['labels'])))
    logger.info('SIZE OF RESULTS: {}'.format(len(all)))

    for label, logit, group, pred, name, trip, in zip(all['labels'], all['logits'], all['groups'], all['preds'],
                                                      all['names'], all['trips']):
        total += 1
        h, r, t = trip.split('\t')
        rel_type = index_to_rel_types[r]
        rel_name_idx = str(rel_to_idx_names[rel_type])
        trip_id = '\t'.join([h, rel_name_idx, t])
        # print('TRIP TEXT: {}'.format(trip_id))
        # print('LOWER 80 SAMPLE: {}'.format(list(lower_80_ids)[0]))
        if upper_20:
            if trip_id not in lower_80_ids:  # NOT = UPPER
                kept += 1
                lower_80_results['labels'].append(label)
                lower_80_results['logits'].append(logit)
                lower_80_results['groups'].append(group)  # groups
                lower_80_results['preds'].append(pred)
                lower_80_results['names'].append(name)
                lower_80_results['trips'].append(trip)
        else:
            if trip_id in lower_80_ids:  # LOWER
                kept += 1
                lower_80_results['labels'].append(label)
                lower_80_results['logits'].append(logit)
                lower_80_results['groups'].append(group)  # groups
                lower_80_results['preds'].append(pred)
                lower_80_results['names'].append(name)
                lower_80_results['trips'].append(trip)
    return lower_80_results, kept, total


def evaluate_test(model, model_dir, set_type="test", eval_lower_80=False, load_eval=False, eval_upper_20=False,
                  run_label=''):
    eval_dataset = load_dataset(set_type, logger, ent_types=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval
    logger.info("***** Running evaluation {} *****".format(set_type))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    result_tracker = collections.defaultdict(list)
    eval_logits, eval_labels, eval_preds, eval_groups, eval_names, eval_trips = [], [], [], [], [], []
    trip_sent_count = collections.defaultdict(int)

    if load_eval:
        # Load evaluation raw data
        fname = os.path.join(model_dir, set_type + "_raw_eval_data.pkl")
        logger.info("Using model: {}".format(model_dir))
        logger.info("Loading raw results file: {}".format(fname))
        with open(fname, "rb") as wf:
            eval = pickle.load(wf)
        eval_loss = eval['loss']
        total_trips = 0
        for label, logit, pred, group, names in zip(eval['labels'], eval['logits'], eval['preds'], eval['groups'],
                                                    eval['names']):
            r = label.item()
            one_trip_per_bag = set()
            h,t = names[0].item(), names[1].item()
            g_h, g_t = group[0][0].item(), group[0][1].item()
            trip = "\t".join([str(h), str(r), str(t)])
            group_and_trip = "\t".join([str(g_h),str(g_t), str(h), str(r), str(t)])
            # Eval distinct trip names in each eval group to compare to eval names experiment
            # Results 1 pos/neg per distinct trip name
            if group_and_trip not in one_trip_per_bag:
                one_trip_per_bag.add(group_and_trip)
                total_trips += 1
                trip_sent_count[trip] += 1
                result_tracker['labels'].append(label)
                result_tracker['logits'].append(logit)
                result_tracker['groups'].append(group)
                result_tracker['preds'].append(pred)
                result_tracker['names'].append(names)
                result_tracker['trips'].append(trip)
        kept_trips = total_trips
        if eval_lower_80:
            result_tracker, kept_trips, total_trips = long_tail_split(result_tracker)
        elif eval_upper_20:
            result_tracker, kept_trips, total_trips = long_tail_split(result_tracker, upper_20=True)
        logger.info("Kept {} trips of {} total trips. Percent: {}%.".format(kept_trips, total_trips,
                                                                            (kept_trips / total_trips)))
        logger.info("Length of labels: {}.".format(len(result_tracker['labels'])))
        eval = {
            'loss': eval_loss,
            'labels': torch.stack(result_tracker['labels']),  # B,
            'logits': torch.stack(result_tracker['logits']),  # B x C
            'groups': torch.stack(result_tracker['groups']),
            'preds': np.asarray(result_tracker['preds']),
            'names': torch.stack(result_tracker['names'])
        }
    else:  # If not loading saved results, run model inference
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
                tmp_eval_loss, logits = model(**inputs)
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

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
            'labels': torch.stack(eval_labels),
            'logits': torch.stack(eval_logits),
            'preds': np.asarray(eval_preds),
            'groups': torch.stack(eval_groups),
            'names': torch.stack(eval_names)
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
    results["loss"] = eval_loss
    logger.info("Results: %s", results)

    if load_eval:
        # Save evaluation results
        with open(os.path.join(model_dir, set_type + "_metrics_from_load_{}.txt".format(run_label)), "w") as wf:
            json.dump(results, wf, indent=4)

    else:
        # Save evaluation results
        with open(os.path.join(model_dir, set_type + "_metrics.txt"), "w") as wf:
            json.dump(results, wf, indent=4)

        # Save evaluation raw data
        with open(os.path.join(model_dir, set_type + "_raw_eval_data.pkl"), "wb") as wf:
            pickle.dump(eval, wf)


def main():
    # Get number of relations
    num_labels = len(read_relations(config.relations_file_types))

    # Load model
    model_dir = '[insert model dir here]'
    logger.info("Evaluate the checkpoint: %s", model_dir)
    model = BertForDistantRE(BertConfig.from_pretrained(model_dir), num_labels, config, bag_attn=config.use_bag_attn)
    model.load_state_dict(torch.load(model_dir + "/pytorch_model.bin", map_location=torch.device(config.device)))
    model.to(config.device)

    # Load raw results (don't re-run model inference)
    load_eval = True

    # Run full set, lower 80 and upper 20
    runs = ['FULL_SET'] #, 'LOWER_80', 'UPPER_20']
    for run_label in runs:

        # Vars to evaluate all data / Lower 80 (rare triples) / Upper 20 (common triples)
        eval_lower_80 = False
        eval_upper_20 = False

        if run_label == 'LOWER_80':
            eval_lower_80 = True
        elif run_label == 'UPPER_20':
            eval_upper_20 = True

        # Evaluation
        evaluate_test(model, model_dir, "test", eval_lower_80=eval_lower_80, load_eval=load_eval,
                      eval_upper_20=eval_upper_20, run_label=run_label)


if __name__ == "__main__":
    main()
