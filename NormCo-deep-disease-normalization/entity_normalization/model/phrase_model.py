import math
import random
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
from smart_open import smart_open
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import entity_normalization.model.scoring as scoring
from utils import text_processing
from entity_normalization.model.prepare_batch import load_text_batch

th.manual_seed(7)
np.random.seed(7)

class CoherenceModel(nn.Module):
    def __init__(self, rnn=nn.GRU, input_dim=200, output_dim=200, dropout_prob=0.3):
        super(CoherenceModel, self).__init__()

        self.input_dim = input_dim        

        self.rnn_dim = output_dim // 2
        self.rnn = rnn(input_dim, self.rnn_dim, batch_first=True, bidirectional=True)

        self.output_dim = output_dim
        self.do = nn.Dropout(p=dropout_prob)

        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initializers
        '''
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_mentions, lengths):
        '''
            input_mentions: b x sl x e
            lengths: b
        '''
        rnn_rep,_ = self.rnn(input_mentions)
        sl = rnn_rep.size()[1]

        output = rnn_rep
        
        return output.view(-1, sl, self.output_dim)

class SummationModel(nn.Module):
    '''
    Simple summation model
    '''

    def __init__(self, embeddings_init=None, vocab_size=10000, embedding_dim=200, sparse=False):
        super(SummationModel, self).__init__()
        
        #Create the word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        if embeddings_init is not None:
            self.embeddings.weight.data.copy_(th.from_numpy(embeddings_init))

        self._vocab_size = vocab_size
        self._e_dim = embedding_dim
   
    def forward(self, inputs):
        # Forward pass for phrases
        examples = inputs['words']
        seq_len = examples.size()[1]
        word_len = examples.size()[2]

        # Mask the pad tokens
        nonzero = (examples != 0).type(th.FloatTensor)
        embs = self.embeddings(examples.view(-1, word_len)) * nonzero.view(-1, word_len).unsqueeze(2)
        embs = embs.view(-1, seq_len, word_len, self._e_dim) 

        # Sum the embeddings
        example_rep = th.sum(embs, dim=2)
        
        #b x sl x e
        return example_rep

class NormalizationModel(nn.Module):
    '''
    Top level normalization model measuring distance from text to concepts
    '''

    def __init__(self, num_diseases, disease_embeddings_init=None, phrase_embeddings_init=None, distfn=scoring.EuclideanDistance(), rnn=nn.GRU, embedding_dim=200, output_dim=200, dropout_prob=0.0, sparse=False, use_features=False):
        super(NormalizationModel, self).__init__()

        # Phrase embedding model
        self.phrase_model = SummationModel(embeddings_init=phrase_embeddings_init,
                                           vocab_size=phrase_embeddings_init.shape[0],
                                           embedding_dim=phrase_embeddings_init.shape[1],
                                           sparse=sparse)

        # Set the distance function
        self.distfn = distfn

        # Concept embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_diseases, embedding_dim, sparse=sparse, padding_idx=0)
        if disease_embeddings_init is not None:
            self.embeddings.weight.data.copy_(th.from_numpy(disease_embeddings_init))
        else:
           self.embeddings.state_dict()['weight'].uniform_(-1e-4, 1e-4) 

        #Linear layer for phrase model
        self.L = nn.Linear(phrase_embeddings_init.shape[1], output_dim)

        #Coherence model
        self.do = nn.Dropout(p=dropout_prob)
        self.coherence = CoherenceModel(rnn, input_dim=output_dim, output_dim=output_dim)
        
        #Parameter to combine models
        self.alpha = Parameter(th.FloatTensor(1))
        nn.init.constant_(self.alpha, 0.5)

        if use_features:
            self.feature_layer = nn.Linear(5, 5)
            self.f2 = nn.Tanh()
            self.score_layer = nn.Linear(5, 1, bias=False)

        self.output_dim = output_dim
        self.use_features = use_features

        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initializers
        '''
        for name, param in self.L.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                #nn.init.eye_(param)
        if self.use_features:
            for name, param in self.feature_layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            for name, param in self.score_layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, inputs, coherence, joint=True):
        '''
            disease_ids: b x sl x nneg
        '''
        # Get phrase, context, and concept representations
        batch_size = inputs['words'].size()[0]
        wseq_len = inputs['words'].size()[1]
        dseq_len = inputs['disease_ids'].size()[1]
        nneg = inputs['disease_ids'].size()[2]

        phrase_rep = self.phrase_model(inputs)
        disease_embs = self.embeddings(inputs['disease_ids'].view(-1, nneg)).view(-1, dseq_len, nneg, self.embedding_dim)
        
        linear_input = phrase_rep.view(-1, self.phrase_model._e_dim)
        
        # Embed phrase into concept space
        example_rep = self.L(linear_input).view(-1, wseq_len, self.output_dim)
        mention_rep = example_rep

        #Coherence vs mention only
        if coherence:
            coherence_rep = self.coherence(self.do(example_rep), inputs['seq_lens'])
            coherence_scores = self.distfn(coherence_rep.unsqueeze(2), disease_embs)

            #Joint vs separate training
            if joint:
                mention_scores = self.distfn(mention_rep.unsqueeze(2), disease_embs)
                alpha = nn.Sigmoid()(self.alpha.unsqueeze(0).unsqueeze(0))
                distance_scores = alpha*mention_scores + (1-alpha)*coherence_scores
            else:
                distance_scores = coherence_scores
        else:
            mention_scores = self.distfn(mention_rep.unsqueeze(2), disease_embs)
            distance_scores = mention_scores            

        if self.use_features and coherence:
            scores = self.score_layer(self.f2(self.feature_layer(th.cat([-inputs['features'], distance_scores.unsqueeze(-1)], dim=-1).view(-1, 5))))
            return scores.view(-1, dseq_len, nneg)
        else:
            return distance_scores

class PreprocessedDataset(Dataset):
    '''
    Dataset reader (preprocess and transform into tensors)
    '''

    def __init__(self, concept_dict, synonym_file, num_neg, vocab_dict=None, use_features=False):

        # Load the concept, vocab, and character dictionaries
        self.concept_dict = pd.read_csv(concept_dict, sep='\t', header=None, comment='#').fillna('')
        self.nconcepts = self.concept_dict.shape[0]

        # Load the training triples
        self.textdb = np.load(synonym_file)
        self.words = self.textdb['words']
        self.lens = self.textdb['lens']
        self.disease_ids = self.textdb['ids']
        self.seq_lens = self.textdb['seq_lens']
        self.features_flags = False
        if 'usefeatures' in self.textdb:
            self.features_flags = self.textdb['usefeatures']

        self.id_dict = {k:i for i,k in enumerate(self.concept_dict.values[:,1])}
        if vocab_dict:
            self.id_vocab = {i:k for k,i in vocab_dict.items()}
        if use_features:
            self.concept_feats = self.preprocess_concept_features()

        self.concept_vocab = list(self.id_dict.keys())

        # Various options
        self.num_neg = num_neg
        self.use_features = use_features

    def __len__(self):
        return self.words.shape[0]

    def preprocess_concept_features(self):
        print("Creating feature dict...")
        concept_feats = defaultdict(lambda: [set(), set(), set(), list(), list()])
        for i,row in enumerate(self.concept_dict.values):
            syn_toks = [set(text_processing.conceptTokenize(s)) for s in [row[0]] + row[7].split('|')]
            stem_toks = [set(text_processing.stem_and_lemmatize(list(toks), lemmatize=False)) for toks in syn_toks]
            ctoks = set([s for toks in syn_toks for s in toks])
            concept_feats[i][0] = ctoks
            concept_feats[i][1] = set(text_processing.stem_and_lemmatize(ctoks, lemmatize=False))
            concept_feats[i][2] = set([''.join([t[0] for t in toks]) for toks in syn_toks])
            concept_feats[i][3] = syn_toks
            concept_feats[i][4] = stem_toks
        print("Done!")
        return concept_feats

    def __getitem__(self, i):
        '''
        Retrieve and preprocess a single item
        '''
        words = self.words[i]
        lens = self.lens[i]
        disease_ids = list(self.disease_ids[i])
        seq_len = self.seq_lens[i]
        features_flag = True
        if self.features_flags:
            features_flag = self.features_flags[i]

        for j in range(len(disease_ids)):
            k = 0
            negs = []
            while k < self.num_neg:
                neg = np.random.randint(0, self.nconcepts)
                if neg != disease_ids[j][0]:
                    negs.append(neg)
                    k += 1
            disease_ids[j] = np.concatenate([disease_ids[j], np.asarray(negs)])

        features = []
        if self.use_features:
            if features_flag:
                for i,d in enumerate(disease_ids):
                    curr_features = []
                    toks = set([self.id_vocab[j] for j in words[i]])
                    for id in d:
                        feats = self.concept_feats[id]
                        stems = set(text_processing.stem_and_lemmatize(toks, lemmatize=False))

                        tok_overlap = toks & feats[0]
                        stem_overlap= stems & feats[1]
                        
                        curr_features.append(np.asarray([float(len(tok_overlap) > 0), 
                                              float(len(stem_overlap) > 0),
                                              max([float(len(toks & ctoks)) for ctoks in feats[3]]) / len(toks),
                                              max([float(len(stems & cstems)) for cstems in feats[4]]) / len(stems)
                                              ]))
                    features.append(np.asarray(curr_features))
            else:
                features = np.zeros((7,len(disease_ids)))

        # Numpy versions
        return th.from_numpy(words), th.from_numpy(lens), th.from_numpy(np.asarray(disease_ids)), th.from_numpy(np.asarray([seq_len])), th.from_numpy(np.asarray(features)).type(th.FloatTensor)

    @classmethod
    def collate(cls, batch):
        '''
        Stacks all of the exampeles in a batch, converts to pytorch tensors
        '''
        words, lens, disease_ids, seq_lens, features = zip(*batch)
        
        return dict(words=words, lens=lens, disease_ids=disease_ids, seq_lens=seq_lens, features=features)

class PreprocessedFakesDataset(Dataset):
    ''' 
    Dataset reader (preprocess and transform into tensors)
    '''

    def __init__(self, concept_dict, synonym_file, num_neg, vocab_dict=None, use_features=False):

        # Load the concept, vocab, and character dictionaries
        self.concept_dict = pd.read_csv(concept_dict, sep='\t', header=None, comment='#').fillna('')
        self.nconcepts = self.concept_dict.shape[0]

        # Load the training triples
        self.textdb = np.load(synonym_file)
        self.words = self.textdb['words']
        self.lens = self.textdb['lens']
        self.disease_ids = self.textdb['ids']
        self.seq_lens = self.textdb['seq_lens']

        self.id_dict = {k:i for i,k in enumerate(self.concept_dict.values[:,1])}
        if vocab_dict:
            self.id_vocab = {i:k for k,i in vocab_dict.items()}
        if use_features:
            self.concept_feats = self.preprocess_concept_features()

        self.concept_vocab = list(self.id_dict.keys())

        # Various options
        self.num_neg = num_neg
        self.use_features = use_features

    def __len__(self):
        return len(self.disease_ids)

    def preprocess_concept_features(self):
        print("Creating feature dict...")
        concept_feats = defaultdict(lambda: [set(), set(), set(), list(), list()])
        for i,row in enumerate(self.concept_dict.values):
            syn_toks = [set(text_processing.conceptTokenize(s)) for s in [row[0]] + row[7].split('|')]
            stem_toks = [set(text_processing.stem_and_lemmatize(list(toks), lemmatize=False)) for toks in syn_toks]
            ctoks = set([s for toks in syn_toks for s in toks])
            concept_feats[i][0] = ctoks
            concept_feats[i][1] = set(text_processing.stem_and_lemmatize(ctoks, lemmatize=False))
            concept_feats[i][2] = set([''.join([t[0] for t in toks]) for toks in syn_toks])
            concept_feats[i][3] = syn_toks
            concept_feats[i][4] = stem_toks
        print("Done!")
        return concept_feats

    def __getitem__(self, i): 
        '''
        Retrieve and preprocess a single item
        '''
        disease_ids = list(self.disease_ids[i])
        seq_len = self.seq_lens[i]
#Select words from disease IDs
        words = []
        lens = []
        for i in range(seq_len[0]):
            k = np.random.randint(len(self.words[disease_ids[i][0]]))
            words.append(self.words[disease_ids[i][0]][k])
            lens.append(self.lens[disease_ids[i][0]][k])

        for j in range(len(disease_ids)):
            k = 0
            negs = []
            while k < self.num_neg:
                neg = np.random.randint(0, self.nconcepts)
                if neg != disease_ids[j][0]:
                    negs.append(neg)
                    k += 1
            disease_ids[j] = np.concatenate([disease_ids[j], np.asarray(negs)])

        features = []
        if self.use_features:
            for i,d in enumerate(disease_ids):
                curr_features = []
                toks = set([self.id_vocab[j] for j in words[i]])
                for id in d:
                    feats = self.concept_feats[id]
                    stems = set(text_processing.stem_and_lemmatize(toks, lemmatize=False))

                    tok_overlap = toks & feats[0]
                    stem_overlap= stems & feats[1]

                    curr_features.append(np.asarray([float(len(tok_overlap) > 0),
                                          float(len(stem_overlap) > 0),
                                          max([float(len(toks & ctoks)) for ctoks in feats[3]]) / len(toks),
                                          max([float(len(stems & cstems)) for cstems in feats[4]]) / len(stems)
                                          ]))
                features.append(np.asarray(curr_features))

        # Numpy versions
        return th.from_numpy(np.asarray(words)), th.from_numpy(np.asarray(lens)), th.from_numpy(np.asarray(disease_ids)), th.from_numpy(np.asarray([seq_len])), th.from_numpy(np.asarray(features)).type(th.FloatTensor)

    @classmethod
    def collate(cls, batch):
        '''
        Stacks all of the exampeles in a batch, converts to pytorch tensors
        '''
        words, lens, disease_ids, seq_lens, features = zip(*batch)
        return dict(words=words, lens=lens, disease_ids=disease_ids, seq_lens=seq_lens, features=features)
