#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import json

import pandas as pd

import smote_variants
from smote_variants import get_simplex_sampling_oversamplers
from smote_variants.evaluation import evaluate_oversamplers

import common_datasets
from common_datasets.binary_classification import get_filtered_data_loaders, get_data_loaders

logger = logging.getLogger('smote_variants')
logger.setLevel(logging.CRITICAL)

smote_variants.config.suppress_external_warnings(False)
smote_variants.config.suppress_internal_warnings(False)


# In[2]:


oversamplers = get_simplex_sampling_oversamplers(within_simplex_sampling='random',
                                                    n_dim_range=2,
                                                    n_quickest=50)


# In[3]:


oversamplers


# In[3]:


datasets = get_filtered_data_loaders(n_col_bounds=(2, 150),
                                        n_minority_bounds=(10, 10000),
                                        n_bounds=(50, 2500),
                                        n_smallest=50,
                                        sorting='n',
                                        n_from_phenotypes=3)

datasets = [dataset() for dataset in datasets]


# In[4]:


classifiers = [('sklearn.neighbors', 'KNeighborsClassifier', {'algorithm': 'brute', 'n_jobs': 1}),
                ('sklearn.tree', 'DecisionTreeClassifier', {'random_state': 5}),
                ('sklearn.ensemble', 'RandomForestClassifier', {'random_state': 5}),
                ('sklearn.svm', 'SVC', {'random_state': 5, 'probability': True}),
                ('smote_variants.classifiers', 'MLPClassifierWrapper', {'random_state': 5})]


# In[5]:


validator_params = {'n_repeats': 2, 'n_splits': 5, 'random_state': 5}

ss_params = {'within_simplex_sampling': 'deterministic',
             'simplex_sampling': 'deterministic'}

vanilla_params = {'random_state': 5,
                    'n_jobs': 1}

deterministic_params = {'random_state': 5,
                        'ss_params': ss_params,
                        'n_jobs': 1}


# In[2]:


cache_path = os.path.expanduser('~/smote-deterministic')


# In[7]:


# creating oversampler specifications

oversampler_list = [('smote_variants', o.__name__, vanilla_params) for o in oversamplers]
oversampler_deterministic = [('smote_variants', o.__name__, deterministic_params) for o in oversamplers]

all_oversamplers = oversampler_list + oversampler_deterministic


# In[8]:


print(len(all_oversamplers))
print(len(datasets))


# In[9]:


dataset_names = [dataset['name'] for dataset in datasets]

for dataset_name in sorted(dataset_names):
    print(dataset_name)


# In[10]:


results = evaluate_oversamplers(datasets=datasets,
                                oversamplers=all_oversamplers,
                                classifiers=classifiers,
                                scaler=('sklearn.preprocessing', 'StandardScaler', {}),
                                validator_params=validator_params,
                                cache_path=cache_path,
                                parse_results=False,
                                clean_up=None,
                                timeout=180,
                                n_jobs=4)


# In[2]:


datasets = smote_variants.evaluation.datasets_in_cache(cache_path)


# In[3]:


datasets


# In[4]:


all_data = []
for dataset in datasets:
    print(dataset)
    data = smote_variants.evaluation.load_dataset_data(datasets[dataset])
    summary = smote_variants.evaluation.create_summary(data)
    summary.to_csv(os.path.join(datasets[dataset], f'summary_{dataset}.csv'))
    data['dataset'] = data['fold_descriptor'].apply(lambda x: x['name'])
    all_data.append(data[['oversampling_error', 'dataset', 'oversampler', 'oversampling_warning']])


# In[5]:


pdf = pd.concat(all_data).reset_index(drop=True)


# In[6]:


print(pdf.head())


# In[7]:


pdf = pdf.reset_index(drop=True)


# In[8]:


print(len(pdf))


# In[9]:


print(pdf[~pdf['oversampling_error'].isnull()][['oversampling_error', 'dataset', 'oversampler', 'oversampling_warning']].drop_duplicates().values)
