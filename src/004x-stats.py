import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

IS_LOCAL = False

if(IS_LOCAL):
    PATH = "../input/Santander/"
else:
    PATH = "../input/"

train_df = pd.read_csv(PATH + "train.csv.zip")
test_df = pd.read_csv(PATH + "test.csv.zip")

# idx = features = train_df.columns.values[2:202]
# for df in [test_df, train_df]:
#     df['sum'] = df[idx].sum(axis=1)
#     df['min'] = df[idx].min(axis=1)
#     df['max'] = df[idx].max(axis=1)
#     df['mean'] = df[idx].mean(axis=1)
#     df['std'] = df[idx].std(axis=1)
#     df['skew'] = df[idx].skew(axis=1)
#     df['kurt'] = df[idx].kurtosis(axis=1)
#     df['med'] = df[idx].median(axis=1)


# features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
# for feature in features:
#     train_df['r2_' + feature] = np.round(train_df[feature], 2)
#     test_df['r2_' + feature] = np.round(test_df[feature], 2)
#     train_df['r1_' + feature] = np.round(train_df[feature], 1)
#     test_df['r1_' + feature] = np.round(test_df[feature], 1)


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}


folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub_df = pd.DataFrame({"ID_code": test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)
