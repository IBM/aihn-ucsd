
# coding: utf-8

# In[1]:


import calour as ca
from calour.training import plot_scatter
from calour.training import RepeatedSortedStratifiedKFold


# In[2]:


# Import Libraries
import os
import math
import numpy as np
import pandas as pd
import biom
import pickle
import time
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from skbio import DistanceMatrix
from scipy.sparse import *
import scipy
from math import sqrt


# In[3]:


# Import Regression Methods
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# # ## Toggle Dataset, balances
# * '82-soil', #0
#                 'PMI_16s', #1
#                 'malnutrition', #2
#                 'cider', #3
#                 'oral_male', #4
#                 'oral_female', #5
#                 'skin_hand_female', #6
#                 'skin_hand_male', #7
#                 'skin_head_female', #8
#                 'skin_head_male', #9
#                 'gut_AGP_female', #10
#                 'gut_AGP_male', #11
#                 'gut_cantonese_female', #12
#                 'gut_cantonese_male' #13
#  
# --balances stores balances=True, --no-balances stores balances=false
# --balances stores balances=True, --no-balances stores balances=false

# In[5]:



parser = argparse.ArgumentParser()
parser.add_argument("dataset", 
                    help="the dataset you wish to benchmark with", 
                    type=int, choices=range(0, 7))
parser.add_argument('--balances', dest='balances', action='store_true')

parser.add_argument('--no_balances', dest='balances', action='store_false')

parser.add_argument('--others', dest='others', action='store_true')

parser.add_argument('--rf', dest='rf', action='store_true')

parser.add_argument('--gb', dest='gb', action='store_true')

parser.add_argument('--et', dest='et', action='store_true')

parser.add_argument('--mlp', dest='mlp', action='store_true')

parser.add_argument('--xgb', dest='xgb', action='store_true')

parser.add_argument('--lsvr', dest='lsvr', action='store_true')

parser.add_argument('--pls', dest='pls', action='store_true')


parser.set_defaults(balances=False)

parser.set_defaults(others=False)
parser.set_defaults(rf=False)
parser.set_defaults(gb=False)
parser.set_defaults(et=False)
parser.set_defaults(mlp=False)
parser.set_defaults(xgb=False)

args = parser.parse_args()
dataset = args.dataset
balances = args.balances


# Give a name for the output data file, directory prefixes
dir_prefixes = ['82-soil', #0
                'PMI_16s', #1
                'malnutrition', #2
                'cider', #3
                'oral_male', #4
                'oral_female', #5
                'skin_hand_female', #6
                'skin_hand_male', #7
                'skin_head_female', #8
                'skin_head_male', #9
                'gut_AGP_female', #10
                'gut_AGP_male', #11
                'gut_cantonese_female', #12
                'gut_cantonese_male' #13
               ]

dir_prefix = dir_prefixes[dataset]
if not os.path.isdir(dir_prefix): 
        os.mkdir(dir_prefix, mode=0o755)
if(balances):
    biom_fp = ['82-soil/balances.qza',
               'PMI_16s/balances.qza',
               'malnutrition/balances.qza',
               'cider/balances.qza',
               'AGP/balances.qza'
              ]
else:
    biom_fp = ['82-soil/rarefied_20000_filtered_samples_features_frequency_table.biom',
               'PMI_16s/PMI_100nt_deblur1-1-0_rarefied8000.biom',
               'malnutrition/rarefied_8500_filtered_samples_features_frequency_table.biom',
               'cider/cider_150nt_rarefied14500.biom',
    "age_prediction/oral_4014/oral_4014__qiita_host_sex_female__.biom",
    "age_prediction/oral_4014/oral_4014__qiita_host_sex_male__.biom",
    "age_prediction/skin_4168/skin_4168__body_site_hand_qiita_host_sex_female__.biom",
    "age_prediction/skin_4168/skin_4168__body_site_hand_qiita_host_sex_male__.biom",
    "age_prediction/skin_4168/skin_4168__body_site_head_qiita_host_sex_female__.biom",
    "age_prediction/skin_4168/skin_4168__body_site_head_qiita_host_sex_male__.biom",
    "age_prediction/gut_4575/gut_4575_rare__cohort_AGP_sex_female__.biom",
    "age_prediction/gut_4575/gut_4575_rare__cohort_AGP_sex_male__.biom",
    "age_prediction/gut_4575/gut_4575_rare__cohort_cantonese_sex_female__.biom",
    "age_prediction/gut_4575/gut_4575_rare__cohort_cantonese_sex_male__.biom"
              ]

metadata_fp = ['82-soil/20994_analysis_mapping_v3.tsv', 
               'PMI_16s/21159_analysis_mapping.txt',
               'malnutrition/merged_metadata_v3.txt',
               'cider/21291_analysis_mapping.txt',
               "age_prediction/oral_4014/oral_4014_map__qiita_host_sex_female__.txt",
               "age_prediction/oral_4014/oral_4014_map__qiita_host_sex_male__.txt",
               "age_prediction/skin_4168/skin_4168_map__body_site_hand_qiita_host_sex_female__.txt",
               "age_prediction/skin_4168/skin_4168_map__body_site_hand_qiita_host_sex_male__.txt",
               "age_prediction/skin_4168/skin_4168_map__body_site_head_qiita_host_sex_female__.txt",
               "age_prediction/skin_4168/skin_4168_map__body_site_head_qiita_host_sex_male__.txt",
               "age_prediction/gut_4575/gut_4575_rare_map__cohort_AGP_sex_female__.txt",
               "age_prediction/gut_4575/gut_4575_rare_map__cohort_AGP_sex_male__.txt",
               "age_prediction/gut_4575/gut_4575_rare_map__cohort_cantonese_sex_female__.txt",
               "age_prediction/gut_4575/gut_4575_rare_map__cohort_cantonese_sex_male__.txt"
              ]

distmatrix_fp = ['82-soil/beta-q2/',
                 'PMI_16s/beta-q2/',
                 'malnutrition/beta-q2/',
                 'cider/beta-q2/'
                ]


# In[8]:


if(balances): 
    feature_datatype = 'qiime2'
    exp = ca.read_amplicon(biom_fp[dataset], metadata_fp[dataset], 
                       data_file_type='qiime2',
                       min_reads=None, normalize=None)
else: #BIOM table input
    exp = ca.read_amplicon(biom_fp[dataset], metadata_fp[dataset], 
                       min_reads=None, normalize=None)
    #if (dataset!=3): exp = exp.filter_abundance(10)


# ## Modify parameter options by shape of data

# Create logarithmic scales for ranges of parameter options where valid inputs can be 1<->n_features or n_samples

# In[11]:


def get_logscale(end, num):
    scale = np.geomspace(start=1, stop=end-1, num=num)
    scale = list(np.around(scale, decimals=0).astype(int))
    return scale


# In[12]:

n_samples = exp.shape[0]
n_features = exp.shape[1]


# In[13]:


#Logarithmic scales based on n_samples
s_logscale = get_logscale(n_samples, 11)
s_logscale7 = get_logscale(n_samples, 8)

s_logscale.pop()
s_logscale7.pop()


# Why .pop()? n_samples is less than total n_samples due to how we split data into folds, so the last item will never be used. e.g.
# ```
# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 123, n_neighbors = 152
# ```

# In[14]:


#Logarithmic scales based on n_features
f_logscale = get_logscale(n_features, 10)
f_logscale7 = get_logscale(n_features, 7)


# __NOTE__ Trimmed the parameter space of time-intensive regressors (ensemble methods, neural net) with many parameters. Original parameter set is "commented" out using triple quotes.
# min_samples_leaf must be at least 1 or in (0, 0.5], got 0.6100000000000001

# In[15]:


# CPU cores
#Use all cores for parallelization, as runtime is determined separately
cpu = -1 

########## Indicates where sample or feature log scales are used

# KNeighbors: use precomputed weights and different X for KNN
KNN_grids = {'n_neighbors': s_logscale, ##########
             'weights': ['uniform', 'distance'], 
             'algorithm': ['brute'],
             'n_jobs': [cpu],
            } #20

# KNeighbors for use with Distance Matrices
KNNDistance_grids = {'n_neighbors': s_logscale, ##########
             'weights':['uniform','distance'], 
             'algorithm': ['brute'],
             'n_jobs': [cpu],
             'p':[2],
             'metric':['precomputed'],
            } #20

# DecisionTree
DT_grids = {'criterion': ['mse'],
            'splitter': ['best','random'],
            'max_depth': s_logscale + [None], ##########
            'max_features': ['auto', 'sqrt', 'log2'],
            'random_state':[2018]
            } #66

# RandomForest
RF_grids = {'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': f_logscale + ['auto', 'sqrt', 'log2'], ##########
            'max_depth': s_logscale + [None], ##########
            'n_jobs': [cpu],
            'random_state': [2018],
            'bootstrap':[True,False],
            'min_samples_split': list(np.arange(0.01, 1, 0.2)),
            'min_samples_leaf': list(np.arange(0.01, .5, 0.1)) + [1],
           } #8580

# ExtraTrees
ET_grids = {'n_estimators': [50, 500, 1000, 5000],
            'criterion': ['mse'],
            'max_features': f_logscale7 + ['auto', 'sqrt', 'log2'], ##########
            'max_depth': s_logscale7 + [None], ########## 
            'n_jobs': [cpu],
            'random_state': [2018],
            'bootstrap':[True,False],
            'min_samples_split': list(np.arange(0.01, 1, 0.2)),
            'min_samples_leaf': list(np.arange(0.01, .5, 0.1)) + [1],
           } #19200

# GradientBoosting
GB_grids = {'loss' : ['ls', 'lad', 'huber', 'quantile'],
            'alpha' : [1e-3, 1e-2, 1e-1, 0.5,0.9],
            'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
            'n_estimators': [1000,5000],
            'criterion': ['mse'],
            'max_features': f_logscale7 + ['auto', 'sqrt', 'log2'], ##########
            'max_depth': s_logscale7 + [None], ##########
            'random_state': [2018]
            } #12800

# Ridge
Ridge_grids = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,
                         1e-2, 1, 5, 10, 20],
               'fit_intercept': [True],
               'normalize': [True, False],
               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 
                           'sparse_cg', 'sag', 'saga'],
               'random_state': [2018]
              } #140

# Lasso
Lasso_grids = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,
                         1e-2, 1, 5, 10, 20],
               'fit_intercept': [True],
               'normalize': [True, False],
               'random_state': [2018],
               'selection': ['random', 'cyclic']
              } #40

# ElasticNet
EN_grids = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,
                         1e-2, 1, 5, 10, 20],
            'l1_ratio': list(np.arange(0.0, 1.1, 0.1)),
            'fit_intercept': [True],
            'random_state': [2018],
            'selection': ['random', 'cyclic']
           } #200

# Linear SVR
LinearSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                   1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'epsilon':[1e-2, 1e-1, 1e0, 1],
             'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
             'random_state': [2018],
             'gamma':['auto', 100, 10, 1, 1e-4, 1e-2, 1e-3, 
                      1e-4, 1e-5, 1e-6],
             'coef0':[0, 1, 10, 100]
            } #3520

# RBF SVR
RSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                   1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'epsilon':[1e-2, 1e-1, 1e0, 1],
             'kernel':['rbf'],
             'gamma':['auto', 100, 10, 1, 1e-4, 1e-2, 1e-3, 
                      1e-4, 1e-5, 1e-6],
             'coef0':[0, 1, 10, 100]
            } #1760

# Sigmoid SVR
SSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                   1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'epsilon':[1e-2, 1e-1, 1e0, 1],
             'kernel':['sigmoid'],
             'gamma':['auto', 100, 10, 1, 1e-4, 1e-2, 1e-3, 
                      1e-4, 1e-5, 1e-6],
             'coef0':[0, 1, 10, 100]
            }
            #'epsilon':[1e-2, 1e-1, 1e0, 1e1, 1e2]
            # Epsilon >10 causes divide by zero error, 
            # C<=0 causes ValueError: b'C <= 0'
            #1760
            
# PLS Regression
PLS_grids = {'n_components': list(np.arange(1,20)),
             'scale': [True, False],
             'max_iter': [500],
             'tol': [1e-08, 1e-06, 1e-04, 1e-02, 1e-00],
             'copy': [True, False]
            } #400

# XGBoost
XGB_grids = {'max_depth': s_logscale + [None], ##########
             'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
             'n_estimators': [1000,5000],
             'objective': ['reg:linear'],
             'booster': ['gbtree', 'gblinear'],
             'n_jobs': [cpu],
             'gamma': [0, 0.2, 0.5, 1, 3],
             'reg_alpha': [1e-3, 1e-1, 1],
             'reg_lambda': [1e-3, 1e-1, 1],
             'scale_pos_weight': [1],
             'base_score': [0.5],
             'random_state': [2018],
             'silent': [1] #no running messages will be printed
            } #9900

# Multi-layer Perceptron 
MLP_grids = {'hidden_layer_sizes': [(100,),(200,),(100,50),(50,50),(25,25,25)],
             'activation': ['identity', 'logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'alpha': [1e-3, 1e-1, 1, 10, 100],
             'batch_size': ['auto'],
             'max_iter': [50,100,200,400],
             'learning_rate': ['constant'],
             'random_state': [2018,14,2362,3456,24,968,90],
            } #7,680


# In[16]:


reg_names = ["KNeighbors",
         "DecisionTree", 
         "RandomForest",
         "ExtraTrees", 
         "GradientBoosting",
         "Ridge", "Lasso", "ElasticNet", 
         "LinearSVR", "RadialSVR", "SigmoidSVR",
         "PLSRegressor",
         "XGBRegressor",
         "MLPRegressor",
        ]
dm_names = [
    "sokalsneath",
    "correlation",
    "dice",
    "cosine", 
    "chebyshev",
    "jaccard",
    "rogerstanimoto",
    "yule",
    "hamming",
    "euclidean",
    "sokalmichener",
    "canberra",
    "matching",
    "braycurtis",
    "aitchison",
    "russellrao",
    "kulsinski",
    "sqeuclidean",
    "cityblock",
    "weighted_unifrac",
    "unweighted_unifrac",
    "weighted_normalized_unifrac",
    "generalized_unifrac"
    ]

ensemble_names = ["RandomForest",
                  "ExtraTrees", 
                  "GradientBoosting",
                  "XGBRegressor"]

#names = ["KNeighbors", "Ridge", "Lasso", "ElasticNet"]
names = reg_names + dm_names #### reg_names + dm_names

dm_set = set(dm_names) # for easy look-up

# Each regressor and their grid, preserving order given above
regressors = [
    KNeighborsRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    Ridge, Lasso, ElasticNet, 
    LinearSVR, SVR, SVR, PLSRegression,
    XGBRegressor,
    MLPRegressor
]

##regressors = [KNeighborsRegressor,Ridge, Lasso, ElasticNet] ########
regressors += [KNeighborsRegressor] * len(dm_names)  ### += to =

all_param_grids = [
    KNN_grids,
    DT_grids,
    RF_grids,
    ET_grids,
    GB_grids,
    Ridge_grids,
    Lasso_grids,
    EN_grids,
    LinearSVR_grids,
    RSVR_grids,
    SSVR_grids,
    PLS_grids,
    XGB_grids,
    MLP_grids,
]

all_param_grids += [KNNDistance_grids] * len(dm_names)


regFromName = dict(zip(names, regressors))
gridFromName = dict(zip(names, all_param_grids))


# ## Main benchmarking loop

# In[18]:


target = None
#Specify column to predict
if (dataset==0): #82-soil
    target = 'ph'
if (dataset==1):
    target = 'days_since_placement'
if (dataset==2):
    target = 'age'
if (dataset==3):
    target = 'fermentation_day'
if (dataset>=4 and dataset<10):
    target = 'qiita_host_age'
if (dataset>-10):
    target = 'age'


# In[19]:



# ENSURE METADATA TARGET IS TYPE INT
exp.sample_metadata[target] = pd.to_numeric(exp.sample_metadata[target])

# In[ ]:


for reg_idx, (reg, name, grid) in enumerate(zip(regressors, names, all_param_grids)):
    
    is_distmatrix = name in dm_set #Boolean switch for distance-matrix specific code blocks
    
    if is_distmatrix: ##### Use specific X and y for distance matrix benchmarking, not amplicon experiment object
        md = exp.sample_metadata
        dm = DistanceMatrix.read(distmatrix_fp[dataset]+name+'.txt')
        md = md.filter(dm.ids,axis='index')
        dm = dm.filter(md.index, strict=True)
        
        X_dist = dm.data
        y_dist = md[target]
        
    if (name=="PLSRegressor"):
        md = exp.sample_metadata
        X_dist = exp.data.toarray()
        y_dist = md[target]

    # Make directory for this regressor if it does not yet exist
    dir_name = dir_prefix +'/' +dir_prefix + '-' + name
    if not os.path.isdir(dir_name): 
        os.mkdir(dir_name, mode=0o755)

    paramsList = list(ParameterGrid(grid))   
        
    # For each set of parameters, get scores for model across 10 folds
    for param_idx, param in enumerate(paramsList):
        
        # If the benchmark data for this param set doesn't exist, benchmark it 
        if not (os.path.isfile(dir_name+'/'+str(param_idx).zfill(5)+'_predictions.pkl') #####changed from if not
                or os.path.isfile(dir_name+'/'+str(param_idx).zfill(5)+'_fold_rmse.pkl')):
            
            if is_distmatrix or (name=="PLSRegressor"): #If benchmarking distance matrix:
                
                # new splits generator for each set of parameterzs
                if (dataset==2): #Use GroupKFold with Malnutrition dataset (2)
                    splits = GroupKFold(n_splits = 16).split(X = X_dist, y = y_dist, groups = md['Child_ID'])
                else:
                    splits = RepeatedSortedStratifiedKFold(5, 3, random_state=2018).split(X_dist, y_dist)
                    
                ### Start Timing
                start = time.process_time()
                df = pd.DataFrame(columns = ['CV', 'SAMPLE', 'Y_PRED', 'Y_TRUE'])
                cv_idx = 0
                
                CV = []
                Y_PRED = []
                Y_TRUE = []
                Y_IDS = []
                
                for train_index, test_index in splits: #y_classes
                    if is_distmatrix:
                        X_train, X_test = X_dist[train_index], X_dist[list(test_index),:][:,list(train_index)]
                    else:
                        X_train, X_test = X_dist[train_index], X_dist[test_index]
                    y_train, y_test = y_dist[train_index], y_dist[test_index]
                    y_train = np.asarray(y_train, dtype='int')
                    y_test_ids = y_dist.index[test_index] ####
                    
                    #print(y_test_ids)
                    #print(X_train, X_train.shape)
                    #print(y_train, y_train.shape)
                    if is_distmatrix:
                        m = KNeighborsRegressor(**param)
                    else:
                        m = reg(**param)

                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_test)

                    CV.extend([cv_idx] * len(y_pred))
                    Y_PRED.extend(y_pred)
                    Y_TRUE.extend(y_test)
                    Y_IDS.extend(y_test_ids)
                    
                    cv_idx += 1
                  
                df['CV'] = CV
                df['Y_TRUE'] = Y_TRUE
                df['Y_PRED'] = Y_PRED
                df['SAMPLE'] = Y_IDS
                end = time.process_time() 
                
                ### End Timing
            
            else: #All others; Not benchmarking distance matrix
            
                if (dataset==2): #Use GroupKFold with Malnutrition dataset (2)
                    it = exp.regress(target, reg(),
                                     cv = GroupKFold(n_splits = 16).split(X = exp.data, 
                                                                          y = exp.sample_metadata['Age_days'], 
                                                                          groups = exp.sample_metadata['Child_ID']), 
                                     params=[param])
                else:
                    it = exp.regress(target, reg(),
                                     cv = RepeatedSortedStratifiedKFold(5, 3, random_state=2018),
                                     params=[param])

                ### Start Timing
                start = time.process_time()
                df = next(it)
                end = time.process_time()
                ### End Timing
            
            # Predictions-level dataframe, saved by param_idx
            df.to_pickle(dir_name+'/'+str(param_idx).zfill(5)+'_predictions.pkl')
                     
            # Calculate RMSE for each fold in this set
            fold_rmse = pd.DataFrame()
            fold_rmse['RMSE'] = df.groupby('CV').apply(lambda x: np.sqrt(mean_squared_error(x['Y_PRED'].values, x['Y_TRUE'].values)))
            fold_rmse['PARAM'] = [param] * fold_rmse.shape[0]
   
            # Store runtimes for this param set
            param_runtime = end-start
            fold_rmse['RUNTIME'] = [param_runtime] * fold_rmse.shape[0]
            fold_rmse.to_pickle(dir_name+'/'+str(param_idx).zfill(5)+'_fold_rmse.pkl')
            
            print(param_idx)
            print(param)


# ## NULL MODELS
# * Needs one null model per dataset
# * Randomly permute y_true 100 times, and compare each permutation to y_true (RMSE)
# * Series, len=102, [mean, median, RMSE_00, ... RMSE99]
# * Save to pkl for use in large box/violin plot. Plot mean+median as points, RMSE as box/violin

# In[22]:


y_true = exp.sample_metadata[target].values
data = []
index = []

for i in range(0,100):
    index.append('RMSE_'+str(i))
    y_perm = np.random.permutation(y_true)
    data.append(sqrt(mean_squared_error(y_perm, y_true)))
    
data = [np.mean(data), np.median(data)] + data
index = ['MEAN', "MEDIAN"] + index
null_model = pd.Series(data, index)

null_model.to_pickle("NULL_MODEL_"+dir_prefix+".pkl")

