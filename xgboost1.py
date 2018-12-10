# coding: utf-8

#（内容）
#   勾配ブースティング回帰(Gradient Boosting regression)とXGBoostの比較。
#   ボストンハウスのデータ(13種類の指標と住宅価格のデータ）の予測を使用。
#（結果）
#   グリッドサーチ クロスバリデーションを使ったXGBoostモデルは
#   勾配ブースティング回帰(Gradient Boosting regression)と比べて、
#   訓練データを使った場合は損失（誤差）は小さくなるが、(0.2391 < 1.7862)
#   テストデータを使った場合は逆に損失（誤差）が大きくなっている。過学習か。　(7.0277 > 6.5871)

'''
This is based on following.

(1) sklearn_examples.py <https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py>
                    Created on 1 Apr 2015
                    @author: Jamie Hall
                   in XGBoost Python Feature Walkthrough.


(2) scikit learn, Gradient Boosting regression example of
<http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py>

Content:
Demonstrate Gradient Boosting on the Boston housing dataset.
This example fits a Gradient Boosting model with least squares loss and
500 regression trees of depth 4.

Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>

License: BSD 3 clause

Copyright <YEAR> <COPYRIGHT HOLDER>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

'''
#Check version
# python 3.6.4 win32 (64bit)
# windows 10 64bit
# scikit-learn (0.19.1)
# scipy (1.0.0)
# numpy (1.14.0)
# xgboost (0.81)


import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_boston

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

import sub_functions
from sub_functions import show_error_value, predict_by_each_estimator, show_worst, show_feature_importance

# ボストンハウスのデータ(13種類の指標と住宅価格のデータ）を読み込んで、9:1の割合で訓練用とテスト用に分ける
boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


# (1)勾配ブースティング回帰モデル(Gradient Boosting Regressor model)のパラメーターを設定する
n_estimators0= 500     # 推定器(estimator)=決定木(decision tree)の総数、
max_depth0= 4          # 決定木(decision tree)の深さ
learning_rate0 = 0.01  # 各推定器(estimator)=決定木(decision tree)の重み付け
loss0= 'ls'            # 損失評価　最小2乗(least squares loss)
params_eGBR = {'n_estimators': n_estimators0, 'max_depth': max_depth0, 'min_samples_split': 2,
          'learning_rate': learning_rate0, 'loss': loss0}
clf_eGBR = ensemble.GradientBoostingRegressor(**params_eGBR)

# 訓練データを使ってモデルをフィッティングする
clf_eGBR.fit(X_train, y_train)

# 結果の表示
show_error_value(clf_eGBR, X_train, y_train, 'Gradient Boosting Regressor model (y_train)')
show_error_value(clf_eGBR, X_test, y_test, 'Gradient Boosting Regressor model (y_test)')


# (2) XGBoostモデル。  XGBRegressor(回帰モデル)のデフォルトの設定値を使ってフィッティングする
#                      他にXGBClassifier、XGBRankerなどがある。
# 各パラメータの説明は<https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst>に記述がある。
# reg:linear: linear regression  (default)
# rmse: root mean square error   (default when regression)
# mae: mean absolute error
xgb_model = xgb.XGBRegressor(booster='gbtree').fit(X_train, y_train, eval_metric="rmse") 

# show_feature_importance(xgb_model, boston)
# print (xgb_model.get_xgb_params()) # show params

# 結果の表示
show_error_value(xgb_model, X_train, y_train, 'XGBoost model by XGBRegressor default condition (y_train)')
show_error_value(xgb_model, X_test, y_test, 'XGBoost model by XGBRegressor default condition (y_test)')


# (3) XGBoostモデル。 XGBRegressorの設定をグリッドサーチ クロスバリデーション(CV)をして良い条件を見つける。  
#                                        他にRandomizedSearchCVがある。
xgb_model_grid = xgb.XGBRegressor()

# グリッドサーチ クロスバリデーション(CV)で試行するパラメータを羅列する
params_grid_cv= {'max_depth': [2,4,6],
              'learning_rate' : [ 0.1, 0.01],
              'n_estimators': [50,100,200,500]}


clf_grid = GridSearchCV(xgb_model_grid, 
                        params_grid_cv, 
                        cv=2,   # 2分割
                        # scoring='neg_mean_squared_error',
                        verbose=1)  # !set verbose=3 if show per iteration

# グリッドサーチの組み合わせ分のフィッティングの実行
print ('')
print ('enter grid search CV...')
clf_grid.fit(X_train,y_train)

# 一番よいもの
print('best score', clf_grid.best_score_)
print('best params', clf_grid.best_params_)
print('best estimator', clf_grid.best_estimator_)

# 結果の表示
show_error_value(clf_grid, X_train, y_train, 'XGBoost model with Grid search CV (y_train)')
show_error_value(clf_grid, X_test, y_test, 'XGBoost model with Grid search CV (y_test)')

