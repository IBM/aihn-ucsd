import gc
import copy
import operator
import argparse
import pickle
import torch as th
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import Counter as mset
from collections import defaultdict
from itertools import groupby
from nltk.metrics.distance import edit_distance
from sklearn.metrics import average_precision_score

from entity_normalization.model.reader_utils import text_to_batch
from entity_normalization.model.prepare_batch import load_text_batch
from utils import text_processing
from utils.text_processing import conceptTokenize

def dataToTorch(normalizing_data, vocab, maxlen=20):
    '''
    Transforms the given data from text to torch tensors
    '''
    #load normalizing data idxs
    test_words = []
    test_lens = []
    seq_lens = []
    for n in normalizing_data:
        words_curr = []
        lens_curr = []
        
        for m in n[1].split('|'):
            #Get word/char indices and lengths
            words, lens = text_to_batch(m, vocab, maxlen=maxlen)
        
            words_curr.append(words)
            lens_curr.append(lens)
        #Pad 0s for sequence length
        seq_lens.append(Variable(th.from_numpy(np.asarray([len(words_curr)]))))
        length = len(words_curr)
    
        #Turn into pytorch variables
        test_words.append(Variable(th.from_numpy(np.asarray(words_curr))).unsqueeze(0))
        test_lens.append(Variable(th.from_numpy(np.asarray(lens_curr))))
    
    return dict(words=test_words, lens=test_lens, seq_lens=seq_lens)

def load_eval_data(disease_file, labels_file, vocab, banner_tags=None, maxlen=20, features_file=None):
    '''
    Loads the disease dictionary, text data, ground truth, and banner tags
    '''

    #Load disease dictionary
    disease_data = pd.read_csv(disease_file, delimiter="\t", comment="#", header=None).fillna('')
    id_dict = {k:i for i,k in enumerate(disease_data.values[:,1])}

    #Load the ground truth
    with open(labels_file) as f:
        next(f)
        testdata = [l.strip().split('\t') for l in f]
    
    labs = []
    for t in testdata:
        for l in t[0].split('|'):
            labs.append((id_dict[l], int(t[3])))
    
    #Load the banner tags if different from ground truth
    normalizing_data = testdata
    if banner_tags:
        normalizing_data = pd.read_csv(banner_tags, sep='\t').fillna('').values        

    #Normalizing data --> torch tensors

    test = dataToTorch(normalizing_data, vocab, maxlen=maxlen)
    test['features'] = []
    test['disease_ids'] = Variable(th.LongTensor(sorted(list(id_dict.values()))))
    if features_file is not None:
        features = np.load(features_file)['features']
        test['features'] = [th.sparse.FloatTensor(th.LongTensor([[r,c] for r,c in zip(f.row, f.col)]).t(), th.from_numpy(f.data).type(th.FloatTensor), th.Size(f.shape)) for f in features]

    return dict(disease_data=disease_data, labs=labs, normalizing_data=normalizing_data, 
                    test=test)

def getScores(model, test, use_coherence=True):
    '''
    Runs the model on the given test data, returning the distance from the test data to each concept
    '''
    preds = []
    test_words = test['words']
    test_lens = test['lens']
    seq_lens = test['seq_lens']
    disease_ids = test['disease_ids']
    features = test['features']
    
    i = 0
    preds = []
    #Run in batches to prevent memory issues
    for i in tqdm(range(len(test_words)), desc="Evaluation"):
        if len(features) > 0:
            curr_f = features[i].to_dense().view(1, -1, disease_ids.shape[0], 4)
            disease_ids_input = disease_ids.unsqueeze(0).unsqueeze(0).repeat(1, curr_f.shape[1], 1)

            scores = model(dict(words=test_words[i], lens=test_lens[i], seq_lens=seq_lens[i], disease_ids=disease_ids_input, features=curr_f), use_coherence, True)

        else:
            scores = model(dict(words=test_words[i], lens=test_lens[i], seq_lens=seq_lens[i], disease_ids=disease_ids.unsqueeze(0).unsqueeze(0)), use_coherence, True)

        preds.append(scores.squeeze(0).data.numpy())
    

    return preds

def eval(model, disease_data, labs, normalizing_data, test, test_batch=10, logfile=None, use_coherence=True):
    '''
    Micro and macro F1 evaluation
    '''    

    preds = getScores(model, test, use_coherence=use_coherence)
    
    preds = np.vstack(preds)
    predict_idxs = np.argmin(preds, axis=1)
    mins = np.min(preds, axis=1)
    #Compare to ground truth
    predictions = list([list(a) for a in zip(predict_idxs, [m for t in normalizing_data for m in t[1].split('|')],
                           [int(m.split(' ')[0]) for t in normalizing_data for m in t[2].split('|')],
                           [int(m.split(' ')[1]) for t in normalizing_data for m in t[2].split('|')],
                           [int(t[3]) for t in normalizing_data for m in t[1].split('|')], mins)])
    
    predictions_by_abstract = defaultdict(list)
    for k,g in groupby(predictions, operator.itemgetter(4)):
        predictions_by_abstract[k].extend(list(g))

    predictions = [(p[0],pmid) for pmid in predictions_by_abstract for p in predictions_by_abstract[pmid]]

    #Micro averaged
    tp = len(list((mset(labs) & mset(predictions)).elements()))
    fp = len(predictions) - tp
    fn = len(labs) - tp

    print("TP: " + str(tp))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    if tp == 0:
        print("No true positives found")
        return 0.0, 0.0
    prec_mic = float(tp) / (tp + fp)
    recall_mic = float(tp) / (tp + fn)
    f_mic = 2 * prec_mic * recall_mic / (prec_mic + recall_mic)
    
    #Macro averaged
    prec_mac = []
    rec_mac = []
    f1_mac = []
    labs_by_abstract = defaultdict(list)
    for k,g in groupby(labs, operator.itemgetter(1)):
        labs_by_abstract[k].extend(list(g))
    predictions_by_abstract = defaultdict(list)
    for k,g in groupby(predictions, operator.itemgetter(1)):
        predictions_by_abstract[k].extend(list(g))
    for k in list(set(predictions_by_abstract.keys()) | set(labs_by_abstract.keys())):
        lab = labs_by_abstract[k]
        pred = predictions_by_abstract[k]
    
        tp = len(list((mset(lab) & mset(pred)).elements()))
        fp = len(pred) - tp
        fn = len(lab) - tp
        prec = 1.0
        if tp + fp > 0:
            prec = float(tp) / (tp + fp)
        rec = 1.0
        if tp + fn > 0:
            rec = float(tp) / (tp + fn)
        f1 = 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
    
        #print("%s\ttp: %d\tfp: %d\tfn: %d\tP: %f\tR: %f\tF1: %f" % (str(k), tp, fp, fn, prec, rec, f1))
    
        prec_mac.append(prec)
        rec_mac.append(rec)
        f1_mac.append(f1)

    print("Macro averaged: P: %f\tR: %f\tF1: %f" %(np.mean(prec_mac), np.mean(rec_mac), np.mean(f1_mac)))
    print("Micro averaged: P: %f\tR: %f\tF1: %f\n" %(prec_mic, recall_mic, f_mic))
    if logfile:
        #logfile.write("Macro averaged: P: %f\tR: %f\tF1: %f\n" %(np.mean(prec_mac), np.mean(rec_mac), np.mean(f1_mac)))
        logfile.write("Micro averaged: P: %f\tR: %f\tF1: %f\n" %(prec_mic, recall_mic, f_mic))

    gc.collect()

    return f_mic#, np.mean(f1_mac)

def accuracy(model, torch_testdata, labs, id_dict, testdata, disease_dict, error_file=None, logfile=None, minimize=True, use_coherence=True, return_errors=False):
    '''
    Evaluate accuracy and ancestor accuracy on ground truth (perfect tagger) as well as MAP and
    MR mapping text into the concept space
    '''

    dict_id = {v:k for k,v in id_dict.items()}
    #Get label accuracy and hierarchy accuracy
    preds = getScores(model, torch_testdata, use_coherence=use_coherence)

    preds = np.vstack(preds)
    predict_idxs = np.argmin(preds, axis=1)
    mins = np.min(preds, axis=1)

    #Group data by abstract for potential post processing
    predictions = list([list(a) for a in zip(predict_idxs, [m for t in testdata for m in t[1].split('|')],
                           [int(m.split(' ')[0]) for t in testdata for m in t[2].split('|')],
                           [int(m.split(' ')[1]) for t in testdata for m in t[2].split('|')],
                           [int(t[3]) for t in testdata for m in t[1].split('|')], mins, np.asarray([l[0] for l in labs]),
                           [m for t in testdata for m in t[0].split('|')])])
    
    predictions_by_abstract = defaultdict(list)
    for k,g in groupby(predictions, operator.itemgetter(4)):
        predictions_by_abstract[k].extend(list(g))
    
    pred_labs = [[p[0], p[6], p[7], p[1], p[4], p[2], p[3]] for pmid in predictions_by_abstract for p in predictions_by_abstract[pmid]]
    predict_idxs = np.asarray([p[0] for p in pred_labs])

    lab_idxs = np.asarray([p[1] for p in pred_labs])
    testdata_singles = [[p[2], p[3], ' ', p[4], p[5], p[6]] for p in pred_labs]

    baseline_accuracy = np.sum(np.equal(predict_idxs, lab_idxs).astype(np.float32)) / len(labs)

    print("Accuracy@1: %f"%baseline_accuracy)
    if error_file is not None or return_errors:
        errors = [['text', 'prediction', 'prediction_preferred', 'actual', 'actual_preferred', 'pmid', 'span_start', 'span_end']]

        for i, (pred,lab) in enumerate(zip(predict_idxs, lab_idxs)):
            errors.append([testdata_singles[i][1], id_dict[pred], disease_dict.values[pred,0], 
                            id_dict[lab], disease_dict.values[lab,0], testdata_singles[i][3], 
                            testdata_singles[i][4], testdata_singles[i][5]])
        df = pd.DataFrame.from_records(errors[1:], columns=errors[0])
    
    if error_file is not None:
        df.to_csv(error_file, sep='\t', index=False)
        
    if logfile:
        logfile.write("Accuracy: %f\n" %(baseline_accuracy))

    if return_errors:
        return df
    else:
        return baseline_accuracy

def getLCAStatistics(disease_dict, tree_map, errors):
    n_errors = 0
    n_child_guessed = 0
    n_ancestor_guessed = 0
    distances = []
    family_errors = []
    lcas = []
    vs = []
    for e in errors.values:
        #Within family errors
        if e[3] != e[5]:
            n_errors += 1
            if e[3] in tree_map[e[5]]:
                if tree_map[e[5]][e[3]] > 0:
                    n_ancestor_guessed += 1
                else:
                    n_child_guessed += 1
                distances.append(abs(tree_map[e[5]][e[3]]))
                family_errors.append((e[3], e[5], tree_map[e[5]][e[3]]))
            
        #LCA distances
        family_gt = tree_map[e[3]]
        family_pred = tree_map[e[5]]
        overlap = family_gt.keys() & family_pred.keys()
        lca = -float('inf')
        dpred = -float('inf')
        dgt = -float('inf')
        ancestor = 'MESH:C'
        #print(overlap)
        for k in overlap:
            if family_gt[k] <= 0 and family_pred[k] <= 0:
                if family_gt[k] + family_pred[k] > lca:
                    lca = family_gt[k] + family_pred[k]
                    dpred = family_pred[k]
                    dgt = family_gt[k]
                    ancestor = k
        v = list(e)
        v.append(ancestor)
        v.append(dgt)
        v.append(dpred)
        if e[3] in family_pred:
            v.append(1)
        else:
            v.append(0)
        vs.append(v)
        lcas.append(lca)
        if lca == -float('inf'):
            print(v)
    
    print("Precent of errors within family: %.03f"%(len(distances) / n_errors))
    print("Average distance within family: %.03f"%(sum(distances) / len(distances)))
    print("Percent child guessed: %.03f"%(n_child_guessed / len(distances)))
    print("Percent ancestor guessed: %.03f"%(n_ancestor_guessed / len(distances)))
    dists = np.asarray(distances)
    print("Percentage of errors which are a child or parent: %.04f"%(len(dists[dists == 1]) / len(dists)))
    
    print("Average LCA distance: %.04f"%(abs(sum(lcas) / len(lcas))))

if __name__ == "__main__":
    import torch.nn as nn
    import entity_normalization.model.phrase_model as m
    import entity_normalization.model.scoring as scoring

    ################################################
    # Parse command line arguments
    ################################################
    parser = argparse.ArgumentParser(description="Train the embedding model to embed synonyms close to each other")
    parser.add_argument('--model', type=str, help='The RNN type for coherence',
                        default='GRU', choices=['LSTM', 'GRU'])
    parser.add_argument('--vocab_file', type=str, help='The location of the vocabulary file', required=True)
    parser.add_argument('--embeddings_file', type=str, help='The location of the pretrained embeddings', required=True)
    parser.add_argument('--sequence_len', type=int, help='The sequence length for phrases', default=20)
    parser.add_argument('--output_dim', type=int, help='The output dimensionality', default=200)
    parser.add_argument('--scoring_type', type=str, help='The type of scoring function to use', default="euclidean", choices=['euclidean', 'bilinear'])
    parser.add_argument('--weight_init', type=str, help='Weights file to initialize the model', required=True)
    parser.add_argument('--disease_dict', type=str, help='The location of the disease dictionary', default=None)
    parser.add_argument('--labels_file', type=str, help='Labels file for inline evaluation', default=None)
    parser.add_argument('--labels_with_abbs_file', type=str, help='Labels file for inline evaluation containing abbreviations', default=None)
    parser.add_argument('--labels_features', type=str, help='Preprocessed features for the labels file', default=None)
    parser.add_argument('--hierarchy_file', type=str, help='File containing the hierarchy of the ontology', default=None)
    parser.add_argument('--html_labels_file', type=str, help='Labels file for visualization', default=None)
    parser.add_argument('--banner_tags', type=str, help='Banner tagged documents for inline evaluation', default=None)
    parser.add_argument('--banner_features', type=str, help='Preprocessed features for the banner tags', default=None)
    parser.add_argument('--error_file', type=str, help='Location of output error analysis file', default=None)
    parser.add_argument('--mention_only', action='store_true', help='Whether or not to use mentions only', default=False)
    
    args = parser.parse_args()
    vocab_dict = text_processing.load_dict_from_vocab_file(args.vocab_file)

    print("Loading evaluation data...")
    eval_data = load_eval_data(args.disease_dict, args.labels_file, vocab_dict, 
                        args.banner_tags, args.sequence_len, features_file=args.banner_features if args.banner_features else args.labels_features)
    print("Done!")

    embeddings_init = np.load(args.embeddings_file)
    
    if args.model in "GRU":
        rnn = nn.GRU
    elif args.model in "LSTM":
        rnn = nn.LSTM

    embedding_dim = args.output_dim
    output_dim = args.output_dim
    
    #Pick the distance function
    if args.scoring_type in "euclidean":
        distance_fn = scoring.EuclideanDistance()
    elif args.scoring_type in "bilinear":
        distance_fn = scoring.BilinearMap(args.output_dim)
        minimize = False
    
    #Create the main model
    model = m.NormalizationModel(eval_data['disease_data'].shape[0],
                            phrase_embeddings_init=embeddings_init,
                            distfn=distance_fn,
                            rnn=rnn, embedding_dim=embedding_dim, output_dim=output_dim, 
                            use_features=(args.labels_features is not None))

    load_state = th.load(args.weight_init)
    state = model.state_dict()
    final_state = {}
    for k in load_state:
        if k in state:
            final_state[k] = load_state[k]
    state.update(final_state)
    model.load_state_dict(state)
    model.eval()

    #Run each round of evaluation

    print("EVALUATING F1 ON TAGS")
    eval(model, **eval_data, use_coherence=not args.mention_only)
    
    print("\nEvaluating accuracy")
    id_dict = {i:k for i,k in enumerate(eval_data['disease_data'].values[:,1])}
    with open(args.labels_file) as f:
        next(f)
        testdata = [l.strip().split('\t') for l in f]
    acc_data = dataToTorch(testdata, vocab_dict, maxlen=args.sequence_len)
    acc_data['disease_ids'] = eval_data['test']['disease_ids']
    acc_data['features'] = []
    if args.labels_features is not None:
        features = np.load(args.labels_features)['features']
        acc_data['features'] = [th.sparse.FloatTensor(th.LongTensor([[r,c] for r,c in zip(f.row, f.col)]).t(), th.from_numpy(f.data).type(th.FloatTensor), th.Size(f.shape)) for f in features]
    errors = accuracy(model, acc_data, eval_data['labs'], id_dict, testdata, eval_data['disease_data'], error_file=args.error_file, use_coherence=not args.mention_only, return_errors=True)

    #For labels with abbreviations
    if args.labels_with_abbs_file is not None:
        with open(args.labels_with_abbs_file) as f:
            next(f)
            testdata = [l.strip().split('\t') for l in f]
        acc_data = dataToTorch(testdata, vocab_dict, maxlen=args.sequence_len)
        acc_data['disease_ids'] = eval_data['test']['disease_ids']
        acc_data['features'] = []
        
        accuracy(model, acc_data, eval_data['labs'], id_dict, testdata, eval_data['disease_data'], use_coherence=not args.mention_only)

    # For LCA distance
    if args.hierarchy_file is not None:
        tree_map = defaultdict(dict)
        with open(args.hierarchy_file) as f:
            for l in f:
                fields = l[:-1].split('\t')
                tree_map[fields[0]][fields[1]] = int(fields[2])
        aggregated_errors = errors.reindex(columns=['pmid', 'span_start', 'span_end', 'actual', 'actual_preferred', 'prediction', 'prediction_preferred'])
        getLCAStatistics(eval_data['disease_data'], tree_map, aggregated_errors)

    #For visualizaing the errors 
    if args.html_labels_file is not None:
        with open(args.html_labels_file) as f:
            next(f)
            testdata = [l.strip().split('\t') for l in f]
        acc_data = dataToTorch(testdata, vocab_dict, maxlen=args.sequence_len)
        acc_data['disease_ids'] = eval_data['test']['disease_ids']
        acc_data['features'] = []
        labs = []
        dict_id = {k:v for v,k in id_dict.items()}
        for t in testdata:
            for l in t[0].split('|'):
                labs.append((dict_id[l], int(t[3])))

        accuracy(model, acc_data, labs, id_dict, testdata, eval_data['disease_data'], use_coherence=not args.mention_only, error_file=args.error_file)
