#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

train_tr = pd.read_csv('train_transaction.csv')
test_tr = pd.read_csv('test_transaction.csv')
train_id = pd.read_csv('train_identity.csv')
test_id = pd.read_csv('test_identity.csv')

# merge data set
train = pd.merge(train_tr,train_id, on = 'TransactionID',how = 'left')
test = pd.merge(test_tr,test_id,on = 'TransactionID',how = 'left')


# In[ ]:


# Bayesian-optimization
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt
# https://www.kaggle.com/clair14/tutorial-bayesian-optimization

# prepare the dataset
y = train.isFraud
# feature_cols should contain "TransactionID"?    I think yes. but it shouldn't be.
features = [c for c in train.columns if c not in ['isFraud','TransactionID']]
X = train[feature_cols]
X_test = test
target = "isFraud"

# cut tr and val
bayesian_tr_idx, bayesian_val_idx = train_test_split(X,y, test_size = 0.25, random_state = 5)
bayesian_tr_idx = bayesian_tr_idx.index
bayesian_val_idx = bayesian_val_idx.index

#black box LGBM 
def LGB_bayesian(
    #learning_rate,
    num_leaves, 
    bagging_fraction,
    feature_fraction,
    min_child_weight, 
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    

    param = {
              'num_leaves': num_leaves, 
              'min_data_in_leaf': min_data_in_leaf,
              'min_child_weight': min_child_weight,
              'bagging_fraction' : bagging_fraction,
              'feature_fraction' : feature_fraction,
              'learning_rate' : learning_rate,
              'max_depth': max_depth,
              'reg_alpha': reg_alpha,# lambda_l1
              'reg_lambda': reg_lambda, # lambda_l2
              'objective': 'binary',
              'save_binary': True,
              'seed': 1337,
              'feature_fraction_seed': 1337,
              'bagging_seed': 1337,
              'drop_seed': 1337,
              'data_random_seed': 1337,
              'boosting_type': 'gbdt',
              'verbose': 1,
              'is_unbalance': True,
              'boost_from_average': True,
              'metric':'auc'}    
    
    oof = np.zeros(len(train))
    trn_data= lgb.Dataset(train.iloc[bayesian_tr_idx][features].values, label=train.iloc[bayesian_tr_idx][target].values)
    val_data= lgb.Dataset(train.iloc[bayesian_val_idx][features].values, label=train.iloc[bayesian_val_idx][target].values)

    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)
    
    oof[bayesian_val_idx]  = clf.predict(train.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  
    
    score = roc_auc_score(train_df.iloc[bayesian_val_idx][target].values, oof[bayesian_val_idx])

    return score

# Bounded region of parameter space
bounds_LGB = {
    'num_leaves': (31, 500), 
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.01, 0.3),
    'min_child_weight': (0.00001, 0.01),   
    'reg_alpha': (1, 2), 
    'reg_lambda': (1, 2),
    'max_depth':(-1,50),
}
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)


# In[ ]:


init_points = 10
n_iter = 15

'''
Main parameters of this function:
n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
'''

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    #The important thing is that our optimization will maximize the value on function.
    #So if your metric should be smaller the better, don't forget to use negative metric value.  
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[ ]:


LGB_BO.max['target']
LGB_BO.max['params']


# In[ ]:


param_lgb = {
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
        #'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_child_weight': LGB_BO.max['params']['min_child_weight'],
        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'reg_lambda': LGB_BO.max['params']['reg_lambda'],
        'reg_alpha': LGB_BO.max['params']['reg_alpha'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), 
        'objective': 'binary',
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'boosting_type': 'gbdt',
        'verbose': 1,
        'is_unbalance': True,
        'boost_from_average': True,
        'metric':'auc'
    }


# In[ ]:


# baseline
params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }

