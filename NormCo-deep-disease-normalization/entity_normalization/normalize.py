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

def load_eval_data(disease_file, tags_file, vocab, maxlen=20, features_file=None):
    '''
    Loads the disease dictionary, text data, ground truth, and banner tags

    '''

    #Load disease dictionary
    disease_data = pd.read_csv(disease_file, delimiter="\t", comment="#", header=None).fillna('')
    id_dict = {k:i for i,k in enumerate(disease_data.values[:,1])}

    normalizing_data = pd.read_csv(tags_file, sep='\t').fillna('').values        

    #Normalizing data --> torch tensors

    test = dataToTorch(normalizing_data, vocab, maxlen=maxlen)
    test['features'] = []
    test['disease_ids'] = Variable(th.LongTensor(sorted(list(id_dict.values()))))
    if features_file is not None:
        features = np.load(features_file)['features']
        test['features'] = [th.sparse.FloatTensor(th.LongTensor([[r,c] for r,c in zip(f.row, f.col)]).t(), th.from_numpy(f.data).type(th.FloatTensor), th.Size(f.shape)) for f in features]

    return dict(disease_data=disease_data, normalizing_data=normalizing_data, 
                    test=test)

def getScores(model, test):
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
    for i in tqdm(range(len(test_words)), desc="Normalizing"):
        if len(features) > 0:
            curr_f = features[i].to_dense().view(1, -1, disease_ids.shape[0], 4)
            disease_ids_input = disease_ids.unsqueeze(0).unsqueeze(0).repeat(1, curr_f.shape[1], 1)

            scores = model(dict(words=test_words[i], lens=test_lens[i], seq_lens=seq_lens[i], disease_ids=disease_ids_input, features=curr_f), True, True)

        else:
            scores = model(dict(words=test_words[i], lens=test_lens[i], seq_lens=seq_lens[i], disease_ids=disease_ids.unsqueeze(0).unsqueeze(0)), True, True)

        preds.append(scores.squeeze(0).data.numpy())
    

    return preds

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
    parser.add_argument('--tags_file', type=str, help='Input NER tags grouped by document', required=True)
    parser.add_argument('--tags_features', type=str, help='Preprocessed features for the labels file', default=None)
    
    args = parser.parse_args()
    vocab_dict = text_processing.load_dict_from_vocab_file(args.vocab_file)

    eval_data = load_eval_data(args.disease_dict, args.tags_file, vocab_dict, 
                        args.sequence_len, features_file=args.tags_features)

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
                            use_features=(args.tags_features is not None))

    load_state = th.load(args.weight_init)
    state = model.state_dict()
    final_state = {}
    for k in load_state:
        if k in state:
            final_state[k] = load_state[k]
    state.update(final_state)
    model.load_state_dict(state)
    model.eval()

    #Get the predictions
    normalizing_data = eval_data['normalizing_data']
    concept_dict = eval_data['disease_data']
    preds = getScores(model, eval_data['test'])
    
    preds = np.vstack(preds)
    predict_idxs = np.argmin(preds, axis=1)
    predict_ids = [concept_dict.values[i,1] for i in predict_idxs]
    mins = np.min(preds, axis=1)
    predictions = list([list(a) for a in zip(predict_ids, [m for t in normalizing_data for m in t[1].split('|')],
                           [int(m.split(' ')[0]) for t in normalizing_data for m in t[2].split('|')],
                           [int(m.split(' ')[1]) for t in normalizing_data for m in t[2].split('|')],
                           [int(t[3]) for t in normalizing_data for m in t[1].split('|')], mins)])
    
    predictions_by_abstract = defaultdict(list)
    for k,g in groupby(predictions, operator.itemgetter(4)):
        predictions_by_abstract[k].extend(list(g))

    print("ids\tmentions\tspans\tpmid")
    for k in predictions_by_abstract:
        preds = predictions_by_abstract[k]
        ids = [p[0] for p in preds]
        mentions = [p[1] for p in preds]
        spans = [str(p[2]) + ' ' + str(p[3]) for p in preds]
        print("|".join(ids) + '\t' + '|'.join(mentions) + '\t' + '|'.join(spans) + '\t' + str(k))
