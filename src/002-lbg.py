# initial lbg of just train
# local score 0.5798
# kaggle score
# maximize score

import os
import sys  # noqa
from time import time
from pprint import pprint  # noqa
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = '' if is_kaggle else '.zip'


def evaluate(train, test, unique_id, target):

    # binary
    lgb_model = lgb.LGBMClassifier(nthread=4, n_jobs=-1, verbose=-1, objective='binary')

    x_train = train.drop([target, unique_id], axis=1)
    y_train = train[target]

    x_test = test[x_train.columns]

    lgb_model.fit(x_train, y_train)

    train_predictions = lgb_model.predict(x_train)
    test_predictions = lgb_model.predict(x_test)

    train_score = roc_auc_score(y_train, train_predictions)

    return test_predictions, train_score

# --------------------- run


def run():

    unique_id = 'ID_code'
    target = 'target'

    # load data

    train = pd.read_csv(f'../input/train.csv{zipext}')
    test = pd.read_csv(f'../input/test.csv{zipext}')

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    test[target] = test_predictions

    predictions = test[[unique_id, target]]

    predictions.to_csv('submission.csv', index=False)


# -------- main

start_time = time()

run()

print(f'{((time() - start_time) / 60):.0f} mins\a')
