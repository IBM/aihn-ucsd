import gc
import time
import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import entity_normalization.model.scoring as scoring
import entity_normalization.model.phrase_model as m
from eval import load_eval_data
from eval import dataToTorch
from eval import accuracy as eval
from utils import text_processing

th.manual_seed(7)
np.random.seed(7)

def train(mention_loader, dict_loader, coherence_loader, distant_loader, fakes_loader, model, optimizer, loss_fn, log_dir='./tb', n_epochs=100, save_every=1, save_file_name='model.pth', eval_data=None, eval_every=10, logfile=None, use_coherence=True):
    '''
    Main training loop
    '''

    # Set mode to training
    model.train()
    step = 0

    # Keep track of best results for far
    acc_best = (-1, 0.0)
    patience = 15
    # Training loop
    for e in range(n_epochs):
        # Evaluate
        if eval_every > 0 and (e + 1) % eval_every == 0:
            model.eval()
            f = None
            if logfile:
                f = open(logfile, 'a')
                f.write("Epoch %d\n"%e)
            acc = eval(model, **eval_data, logfile=f)
            if logfile:
                f.write('\n')
                f.close()
            if acc >= acc_best[1]:
                acc_best = (e, acc)
                th.save(model.state_dict(),save_file_name + "_bestacc_%05d"%(e))
            elif e - acc_best[0] > patience:
                #Early stopping
                break  

            model.train()
            gc.collect()

        # Epoch loops
        #Dictionary data
        for mb in tqdm(dict_loader, desc='Epoch %d'%e):
            mb['words'] = Variable(th.stack(mb['words'], 0))
            mb['lens'] = Variable(th.cat(mb['lens'], 0))
            mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
            mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
            if model.use_features:
                mb['features'] = Variable(th.stack(mb['features'], 0))

            # Mention step
            optimizer.zero_grad()
            #Pass through the model
            mention_scores = model(mb, False)
            #Get sequence length and number of negatives
            nneg = mention_scores.size()[2]
            scores = mention_scores
            #Get the loss
            loss = loss_fn(scores.view(-1, nneg))
            loss.backward(retain_graph=True)
            optimizer.step()

            step += 1
        #Mention data
        for mb in tqdm(mention_loader, desc='Epoch %d'%e):
            mb['words'] = Variable(th.stack(mb['words'], 0))
            mb['lens'] = Variable(th.cat(mb['lens'], 0))
            mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
            mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
            if model.use_features:
                mb['features'] = Variable(th.stack(mb['features'], 0))

            # Mention step
            optimizer.zero_grad()
            #Pass through the model
            mention_scores = model(mb, False)
            #Get sequence length and number of negatives
            nneg = mention_scores.size()[2]
            scores = mention_scores
            #Get the loss
            loss = loss_fn(scores.view(-1, nneg))
            loss.backward(retain_graph=True)
            optimizer.step()

            step += 1
        #Fake data
        for fb in tqdm(fakes_loader, desc='Epoch %d'%e):

            fb['words'] = Variable(th.stack(fb['words'], 0))
            fb['lens'] = Variable(th.cat(fb['lens'], 0))
            fb['disease_ids'] = Variable(th.stack(fb['disease_ids'], 0))
            fb['seq_lens'] = Variable(th.cat(fb['seq_lens'], 0))
            if model.use_features:
                fb['features'] = Variable(th.stack(fb['features'], 0))
            # Coherence step
            optimizer.zero_grad()
            #Pass through the model
            scores = model(fb, use_coherence)
            #Get sequence length and number of negatives
            nneg = scores.size()[2]
            #Get the loss
            loss = loss_fn(scores.view(-1, nneg))
            loss.backward(retain_graph=True)
            optimizer.step()

            step += 1
        #Distantly supervised data
        for mb in tqdm(distant_loader, desc='Epoch %d'%e):
            mb['words'] = Variable(th.stack(mb['words'], 0))
            mb['lens'] = Variable(th.cat(mb['lens'], 0))
            mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
            mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
            if model.use_features:
                mb['features'] = Variable(th.stack(mb['features'], 0))

            # Mention step
            optimizer.zero_grad()
            #Pass through the model
            mention_scores = model(mb, use_coherence)
            #Get sequence length and number of negatives
            nneg = mention_scores.size()[2]
            scores = mention_scores
            #Get the loss
            loss = loss_fn(scores.view(-1, nneg))
            loss.backward(retain_graph=True)
            optimizer.step()

            step += 1
        #Coherence data
        for cb in tqdm(coherence_loader, desc='Epoch %d'%e):

            cb['words'] = Variable(th.stack(cb['words'], 0))
            cb['lens'] = Variable(th.cat(cb['lens'], 0))
            cb['disease_ids'] = Variable(th.stack(cb['disease_ids'], 0))
            cb['seq_lens'] = Variable(th.cat(cb['seq_lens'], 0))
            if model.use_features:
                cb['features'] = Variable(th.stack(cb['features'], 0))
            # Coherence step
            optimizer.zero_grad()
            #Pass through the model
            scores = model(cb, use_coherence)
            #Get sequence length and number of negatives
            nneg = scores.size()[2]
            #Get the loss
            loss = loss_fn(scores.view(-1, nneg))
            loss.backward(retain_graph=True)
            optimizer.step()

            step += 1

        gc.collect()
    
    # Log final best values
    if logfile:
        with open(logfile, 'a') as f:
            f.write("Best accuracy: %f in epoch %d\n"%(acc_best[1], acc_best[0]))

if __name__ == "__main__":
    ################################################
    # Parse command line arguments
    ################################################
    parser = argparse.ArgumentParser(description="Train the embedding model to embed synonyms close to each other")
    parser.add_argument('--model', type=str, help='The RNN type for coherence',
                        default='GRU', choices=['LSTM', 'GRU'])
    parser.add_argument('--vocab_file', type=str, help='The location of the vocabulary file', required=True)
    parser.add_argument('--embeddings_file', type=str, help='The location of the pretrained embeddings', required=True)
    parser.add_argument('--disease_embeddings_file', type=str, help='The location of pretrained disease embeddings', default=None)
    parser.add_argument('--train_data', type=str, help='The location of the training data', default=None)
    parser.add_argument('--dictionary_data', type=str, help='The location of the dictionary mentions', default=None)
    parser.add_argument('--distant_data', type=str, help='The location of the distantly supervised data', default=None)
    parser.add_argument('--coherence_data', type=str, help='The location of the coherence data', required=True)
    parser.add_argument('--fake_data', type=str, help='The location of synthetic data to train the coherence model', default=None)
    parser.add_argument('--num_epochs', type=int, help='The number of epochs to run', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size for mini batching', default=32)
    parser.add_argument('--sequence_len', type=int, help='The sequence length for phrases', default=20)
    parser.add_argument('--num_neg', type=int, help='The number of negative examples', default=1)
    parser.add_argument('--output_dim', type=int, help='The output dimensionality', default=200)
    parser.add_argument('--lr', type=float, help='The starting learning rate', default=0.001)
    parser.add_argument('--l2reg', type=float, help='L2 weight decay', default=0.0)
    parser.add_argument('--dropout_prob', type=float, help='Dropout probability', default=0.0)
    parser.add_argument('--scoring_type', type=str, help='The type of scoring function to use', default="euclidean", choices=['euclidean', 'bilinear', 'cosine'])
    parser.add_argument('--weight_init', type=str, help='Weights file to initialize the model', default=None)
    parser.add_argument('--threads', type=int, help='Number of parallel threads to run', default=1)
    parser.add_argument('--save_every', type=int, help='Number of epochs between each model save', default=1)
    parser.add_argument('--save_file_name', type=str, help='Name of file to save model to', default='model.pth')
    parser.add_argument('--optimizer', type=str, help='Which optimizer to use', default='sgd', choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'])
    parser.add_argument('--loss', type=str, help='Which loss function to use', default='maxmargin', choices=['maxmargin', 'xent'])
    parser.add_argument('--eval_every', type=int, help='Number of epochs between each evaluation', default=0)
    parser.add_argument('--disease_dict', type=str, help='The location of the disease dictionary', default=None)
    parser.add_argument('--labels_file', type=str, help='Labels file for inline evaluation', default=None)
    parser.add_argument('--banner_tags', type=str, help='Banner tagged documents for inline evaluation', default=None)
    parser.add_argument('--test_features_file', type=str, help='File containing test features when features are used', default=None)
    parser.add_argument('--use_features', action='store_true', help='Whether or not to use hand crafted features', default=False)
    parser.add_argument('--mention_only', action='store_true', help='Whether or not to use mentions only', default=False)
    parser.add_argument('--logfile', type=str, help='File to log evaluation in', default=None)
    
    args = parser.parse_args()
    print(args)
    
    vocab_dict = text_processing.load_dict_from_vocab_file(args.vocab_file)
    # First set up the dataset
    if args.train_data is not None:
        mention_data = m.PreprocessedDataset(args.disease_dict, args.train_data, args.num_neg, vocab_dict, args.use_features)
        mention_loader = DataLoader(
            mention_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.threads,
            collate_fn=mention_data.collate
        )
    else:
        mention_loader = []
    if args.dictionary_data is not None:
        dictionary_data = m.PreprocessedDataset(args.disease_dict, args.dictionary_data, args.num_neg, vocab_dict, args.use_features)
        dict_loader = DataLoader(
            dictionary_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.threads,
            collate_fn=mention_data.collate
        )
    else:
        dict_loader = []
    if args.distant_data is not None:
        distant_data = m.PreprocessedDataset(args.disease_dict, args.distant_data, args.num_neg, vocab_dict, args.use_features)
        distant_loader = DataLoader(
            distant_data,
            batch_size=1,
            shuffle=True,
            num_workers=args.threads,
            collate_fn=mention_data.collate
        )
    else:
        distant_loader = []
    if args.fake_data is not None:
        fake_data = m.PreprocessedFakesDataset(args.disease_dict, args.fake_data, args.num_neg, vocab_dict, args.use_features)
        fakes_loader = DataLoader(
            fake_data,
            batch_size=1,
            shuffle=True,
            num_workers=args.threads,
            collate_fn=mention_data.collate
        )
    else:
        fakes_loader = []
    
    coherence_data = m.PreprocessedDataset(args.disease_dict, args.coherence_data, args.num_neg, vocab_dict, args.use_features)
    coherence_loader = DataLoader(
        coherence_data,
        batch_size=1,
        shuffle=True,
        num_workers=args.threads,
        collate_fn=coherence_data.collate
    )
    
    # Set up the evaluation dataset for inline evaluation
    eval_data = None
    if args.eval_every > 0:
        print("Loading evaluation data...")
        eval_data = load_eval_data(args.disease_dict, args.labels_file, vocab_dict,
                            args.banner_tags, args.sequence_len,
                            features_file=args.test_features_file)
        id_dict = {i:k for i,k in enumerate(eval_data['disease_data'].values[:,1])}
        with open(args.labels_file) as f:
            next(f)
            testdata = [l.strip().split('\t') for l in f]
        acc_data = dataToTorch(testdata, vocab_dict, maxlen=args.sequence_len)
        acc_data['disease_ids'] = eval_data['test']['disease_ids']
        acc_data['features'] = eval_data['test']['features']
        eval_data = dict(torch_testdata=acc_data, labs=eval_data['labs'], id_dict=id_dict,testdata=testdata,disease_dict=eval_data['disease_data'],error_file=None) 
        print("Done!")
    
    sparse = True
    if args.optimizer in 'adam':
        sparse = False
    
    if args.model in "GRU":
        rnn = nn.GRU
    elif args.model in "LSTM":
        rnn = nn.LSTM
    
    embedding_dim = args.output_dim
    output_dim = args.output_dim

    # Pick the distance function
    margin = np.sqrt(output_dim)
    if args.scoring_type in "euclidean":
        distance_fn = scoring.EuclideanDistance()
    if args.scoring_type in "cosine":
        distance_fn = scoring.CosineSimilarity(dim=-1)
        margin = args.num_neg - 1
    elif args.scoring_type in "bilinear":
        distance_fn = scoring.BilinearMap(output_dim)
        margin = 1.0
    
    # Load concept embeddings initializer
    disease_embs_init = None
    if args.disease_embeddings_file:
        disease_embs_init = np.load(args.disease_embeddings_file)

    # Load initial word embeddings
    embeddings_init = np.load(args.embeddings_file)
    
    # Create the normalization model
    model = m.NormalizationModel(len(coherence_data.id_dict.keys()), 
                            disease_embeddings_init=disease_embs_init, 
                            phrase_embeddings_init=embeddings_init,
                            distfn=distance_fn,
                            rnn=rnn, embedding_dim=embedding_dim, output_dim=output_dim, 
                            dropout_prob=args.dropout_prob, sparse=sparse,
                            use_features=args.use_features)
    
    # Choose the optimizer
    parameters = []
    default_params = []
    for name,param in model.named_parameters():
        if param.requires_grad:
            if name in 'feature_layer.weight' or name in 'score_layer.weight':
                default_params.append(param)
            else:
                parameters.append({'params': param, 'weight_decay': 0.0})
    parameters.append({'params': default_params})
    
    if args.optimizer in 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2reg, momentum=0.9)
    elif args.optimizer in 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.l2reg)
    elif args.optimizer in 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.l2reg)
    elif args.optimizer in 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=args.lr, weight_decay=args.l2reg)
    elif args.optimizer in 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2reg)
    
    # Pick the loss function
    if args.loss in 'maxmargin':
        loss = scoring.MaxMarginLoss(margin=margin)
    elif args.loss in 'xent':
        loss = scoring.CrossEntropyDistanceLoss()
    
    # Load pretrained weights if given
    if args.weight_init:
        model.load_state_dict(th.load(args.weight_init))
    
    # Train!
    train(mention_loader, dict_loader, coherence_loader, distant_loader, fakes_loader, model, optimizer, loss, n_epochs=args.num_epochs, save_every=args.save_every, save_file_name=args.save_file_name, eval_data=eval_data, eval_every=args.eval_every, logfile=args.logfile, use_coherence=not args.mention_only)
