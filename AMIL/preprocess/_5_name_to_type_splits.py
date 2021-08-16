import collections
import json
import pickle

import numpy as np

import config
from utils.utils import JsonlReader


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def random_upsample(last_sent_bag, last_ent_name_bag, bag_size):
    '''Randomly upsample sentences and corresponding ent names in the final bag of sents'''
    idxs = list(np.random.choice(list(range(len(last_sent_bag))), bag_size - len(last_sent_bag)))
    s_bag = last_sent_bag + [last_sent_bag[i] for i in idxs]
    n_bag = last_ent_name_bag + [last_ent_name_bag[i] for i in idxs]
    assert len(n_bag) == bag_size
    assert len(s_bag) == bag_size
    return s_bag, n_bag


def load_txt_to_type():
    with open(config.umls_text_to_type, "rb") as f:
        txt_to_type = pickle.load(f)
    return txt_to_type


def write_set_to_file(in_set, fname):
    with open(fname, mode='wt') as cw:
        cw.write('\n'.join(list(in_set)))
    print('Wrote file: {}'.format(fname))


def write_new_triple_names_file(new_split_trips, triples_fname):
    with open(triples_fname, "w") as wf:
        for trip in list(new_split_trips):
            h, r, t = trip.split('\t')
            h, r, t = h.lower(), r.lower(), t.lower()
            wf.write("{}\t{}\t{}\n".format(h, r, t))
        print('Triple file updated: {}', triples_fname)


def reorg_names_to_types(in_fname, out_fname, triples_fname, trips_type_fname, all_r, all_e):
    complete_types = collections.defaultdict(list)
    txt_to_type = load_txt_to_type()

    jr = list(iter(JsonlReader(in_fname)))
    new_split_trips = set()
    new_split_trips_types = set()
    for idx, jsonl in enumerate(jr):
        rel = jsonl["relation"].lower()
        h, t = jsonl["group"]
        h_type, t_type = list(txt_to_type[h])[0].lower(), list(txt_to_type[t])[0].lower()
        sentences = jsonl["sentences"]
        unique_sents = set()
        new_split_trips.add('\t'.join([h, rel, t]))
        new_split_trips_types.add('\t'.join([h_type, rel, t_type]))

        for sent in sentences:
            if sent not in unique_sents:

                # Track all rels and ent types
                all_r.add(rel)
                all_e.add(h_type)
                all_e.add(t_type)
                unique_sents.add(sent)

                if config.use_orig_sents:
                    key = '\t'.join([h_type, rel, t_type])
                    sent_h_t = "\t".join([sent, h, t])
                    complete_types[key].append(sent_h_t)
                else:
                    # Head name -> type
                    split_h = sent.split('$')  # h
                    split_h[1] = h_type
                    sent = '$'.join(split_h)

                    # Tail name -> type
                    split_t = sent.split('^')  # t
                    split_t[1] = t_type
                    sent_type = '^'.join(split_t)

                    # Quality check
                    assert (len(split_h) == 3)
                    assert (len(split_t) == 3)

                    # New sentence
                    sent_type_h_t = "\t".join([sent_type, h, t])
                    key = '\t'.join([h_type, rel, t_type])
                    complete_types[key].append(sent_type_h_t)

    # Write new triples types file
    write_set_to_file(new_split_trips_types, trips_type_fname)
    write_new_triple_names_file(new_split_trips, triples_fname)

    # Save new complete types data file
    with open(out_fname, "w", encoding="utf-8") as wf:
        for hrt in complete_types:  # for each ent type group
            sent_list, ent_name_list = [], []
            h_type, rel, t_type = hrt.split('\t')
            h_type, rel, t_type = h_type.lower(), rel.lower(), t_type.lower()
            for sent_h_t in complete_types[hrt]:
                sent, h, t = sent_h_t.split('\t')
                h, t = h.lower(), t.lower()
                sent_list.append(sent)
                ent_name_list.append((h, t))

            # Split all sents and ent names into bags of n=bag_size
            bags_sents = list(chunks(sent_list, config.bag_size))
            bags_names = list(chunks(ent_name_list, config.bag_size))

            # Random up-sample the final bag of sents and ent names
            if len(bags_sents[-1]) < config.bag_size:
                bags_sents[-1], bags_names[-1] = random_upsample(bags_sents[-1], bags_names[-1], config.bag_size)

            # Assert all bags are full
            for n_bag, s_bag in zip(bags_names, bags_sents):
                assert (len(s_bag) == config.bag_size)
                assert (len(n_bag) == config.bag_size)

            # Write groups to the jsonl file
            for sent_group, name_group in zip(bags_sents, bags_names):
                jdata = {"group": [h_type, t_type], "relation": rel, "ent_names": name_group,
                         "sentences": sent_group}
                wf.write(json.dumps(jdata) + "\n")
    return all_r, all_e


if __name__ == "__main__":
    all_r, all_e = set(), set()
    all_r, all_e = reorg_names_to_types(config.complete_train, config.complete_types_train,
                                        config.triples_file_train, config.triples_types_file_train, all_r, all_e,
                                        config.remove_na)
    all_r, all_e = reorg_names_to_types(config.complete_dev, config.complete_types_dev, config.triples_file_dev,
                                        config.triples_types_file_dev, all_r, all_e, config.remove_na)
    all_r, all_e = reorg_names_to_types(config.complete_test, config.complete_types_test, config.triples_file_test,
                                        config.triples_types_file_test, all_r, all_e, config.remove_na)

    # Create entities.types.txt
    write_set_to_file(all_e, config.entities_file_types)

    # Create relations.types.txt
    write_set_to_file(all_r, config.relations_file_types)
