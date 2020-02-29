#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
# from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder 
# import multiprocessing
import gc 

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV

train_tr = pd.read_csv('C:\\ieee-fraud-detection\\train_transaction.csv')
test_tr = pd.read_csv('C:\\ieee-fraud-detection\\test_transaction.csv')
train_id = pd.read_csv('C:\\ieee-fraud-detection\\train_identity.csv')
test_id = pd.read_csv('C:\\ieee-fraud-detection\\test_identity.csv')


train=pd.merge(train_tr,train_id,on='TransactionID',how='left')
test=pd.merge(test_tr,test_id,on='TransactionID',how='left')

del test_id, test_tr, train_id, train_tr
gc.collect()

# clean some useless feature
one_value_cols=[col for col in train.columns if train[col].nunique()<=1]
one_value_cols_test=[col for col in test.columns if test[col].nunique()<=1]

many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
# value_counts(dropna=False,normalize=Ture) 对Series里的每个值进行计数，返回一个Series数组 normalize=Ture 可以得出计数占比

cols_to_drop=list(set(many_null_cols+many_null_cols_test+big_top_value_cols+big_top_value_cols_test+one_value_cols+one_value_cols_test))
# set() 创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可计算交集、并集、补集、差集等

cols_to_drop.remove('isFraud')
print('{} features are going to be dropped for being uesless'.format(len(cols_to_drop)))

train = train.drop(cols_to_drop)
test = test.drop(cols_to_drop)

def reduce_men_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # 1024**2 = 1024*1024
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        # values 返回 array 数组
        train[col]=le.transform(list(train[col].astype(str).values))
del test
gc.collect()

train = reduce_men_usage(train)
X = train.sort_values('TranscationDT').drop(['isFarud', 'TranscationDT', 'TransactionID'], axis=1)
y = train.sort_values('TranscationDT')['isFraud']
del train
gc.collect()

# refecv does not support nans
X.fillna(-999, inplace=True)
params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }
clf = lgb.LGBMClassifier(**params)
rfe = RFECV(estimator=clf, step=10, cv=KFold(nsplit=5, shuffle=False), scoring='roc_auc', verbose=2)

# estimator 提供关于特征重要信息的方法
# step 每次迭代式要移除的整数个特征数
# cv 交叉验证生成器

rfe.fit(X, y)

print("Optimal number of features :%d" % rfe.n_features_)


plt.figure(figsize=(14, 8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()

# ranking 特征的排位，估计最佳的属性被排为 1
for col in X.colmns[rfe.ranking_ == 1]:
    # 输出最佳有用特征
    print(col)





