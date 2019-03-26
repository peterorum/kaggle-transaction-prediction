import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = '' if is_kaggle else '.zip'
train_file = 'train'  # if is_kaggle else 'sample'

start_time = time()

unique_id = 'ID_code'
target = 'target'

# load data

train = pd.read_csv(f'../input/{train_file}.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')


print(f'load data, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

fig = plt.figure()

important_cols = [
    'var_81',
    'var_174',
    'var_53',
    'var_139',
    'var_110',
    'var_146',
    'var_6',
    'var_26',
    'var_22',
    'var_80'
]

for col in important_cols:
    fig = plt.figure()

    sb.distplot(train[col].fillna(0), kde=True, bins=30)

    fig.savefig(f'../plots/target-{col}.png')

    plt.close(fig)


print(f'{((time() - start_time) / 60):.0f} mins\a')
