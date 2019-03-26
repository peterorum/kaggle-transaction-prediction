# baseline: constant value
# local score
# kaggle score

import sys  # noqa
# import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from time import time

import os

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = '' if is_kaggle else '.zip'
train_file = 'train' if is_kaggle else 'sample'

# load data
train = pd.read_csv(f'../input/{train_file}.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

# -------- main

start_time = time()

target = 'target'
unique_id = 'ID_code'

# constant
result = 0

train['predicted'] = result

score = roc_auc_score(train[target], train.predicted)
print('score', score)

test[target] = result

# print(test.head())
# print(test.describe())

predictions = test[[unique_id, target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
