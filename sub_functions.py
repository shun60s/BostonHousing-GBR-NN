# coding: utf-8

# GradientBoostingRegressortための機能ファンクションを個々のdefへ分割したもの。

'''
This is based on following scikit learn, Gradient Boosting regression example of
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
# graphviz (0.8.2)
# matplotlib (2.1.1)


import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import export_graphviz


# 各推定器(estimator)=決定木(decision tree)を使って予測値を計算する
def predict_by_each_estimator(clf, X_test, y_train, y_test, params):
    y_predict_by_each_estimator= np.zeros(len(y_test))
    y_predict_by_each_estimator+= np.mean(y_train)  # F0　初期値として訓練データの平均値を設定する
    print ('')
    print ('computing using each estimator step by step')
    for i in range (len (clf.estimators_) ): # 推定器(estimator)毎ごとに計算する（回帰式）
        regressor= clf.estimators_ [i,0]  # 各推定器(estimator)=決定木(decision tree)を取り出す
        yp=regressor.predict(X_test)      # その決定木に従って値（Value）を求める
        y_predict_by_each_estimator += yp * params['learning_rate']  # その値（Value）に重みを掛けて、足し合わせる
        # 注意：　値（Value）は固定値である。連続量ではない。従って、扱っている量が連続でも　予測値は離散化された値になる。
    return y_predict_by_each_estimator


# テストデータを使って予測誤差を計算する
# 住宅価格の予想値と実際の価格差の評価のため 誤差の絶対値の平均値MAE(mean absolute error)も計算する
def show_error_value(clf, X_test, y_test, model0='Gradient Boosting Regressor model'):
    mse = mean_squared_error(y_test, clf.predict(X_test))
    mae = mean_absolute_error(y_test, clf.predict(X_test))
    print ('')
    print ('model: ', model0)
    print("MSE(mean square error): %.4f" % mse)   # 6.6853
    print("MAE(mean absolute error): %.4f" % mae) # 1.9192


# 予測誤差が大きいWorst n_thを表示する
def show_worst( clf, X_test, y_test, n_th=10):
    Y_predict= clf.predict(X_test)
    index0= np.argsort( np.abs(y_test - Y_predict))[::-1]
    print ('')
    print ('Worst :', n_th)
    print ('actual, estimation, difference')
    for i in index0[0:n_th]:
        print(y_test[i], "%.4f" % Y_predict[i], "%.4f" % (Y_predict[i] - y_test[i]) )



# 各指標の重要度をプリントアウトする
def show_feature_importance(clf, boston):  # ,X_train, y_train):
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[::-1]
    print ('')
    print ('Feature Importance: Relative Importance')
    for i in sorted_idx:
        print (" " , dict[boston.feature_names[i]], " %.3f " %   feature_importance[i]) # 日本語表示
        #print (" " , boston.feature_names[i], " %.3f " %   feature_importance[i])  # alphabet word
        '''
        # 散布図も描く
        plt.scatter(X_train[:,i],y_train)
        plt.xlabel( boston.feature_names[i])
        plt.ylabel('house price')
        plt.show()
        '''

def write_dot():
    # 各推定器(estimator)=決定木(decision tree)の構成を dotファイルとして書き出す
    dot_dir='./dot'  # dot　ファイルを格納するディレクトリー名称
    print ('')
    print('saving... each estimator decision tree')
    if not os.path.exists(dot_dir):
        os.makedirs(dot_dir)
    for i in range (len (clf.estimators_) ):
        regressor= clf.estimators_ [i,0]
        outfile0= os.path.join( dot_dir, 'tree' + str(i) + '.dot')
        export_graphviz(regressor, out_file= outfile0, feature_names=boston.feature_names)


# 13種類の指標を日本語の表記に変換する辞書　（<http://liaoyuan.hatenablog.jp/entry/2018/02/03/004849>からの引用）
dict= { "LSTAT": "低所得者の割合",
        "RM": "1戸あたりの平均部屋数",
        "DIS": "ボストンの主な5つの雇用圏までの重み付き距離",
        "AGE": "1940年よりも前に建てられた家屋の割合",
        "TAX": "10,000ドルあたりの所得税率",
        "PTRATIO": "教師あたりの生徒の数（人口単位）",
        "CRIM": "犯罪発生率（人口単位）", 
        "B": "黒人居住者の割合（人口単位）",
        "NOX": "窒素酸化物の濃度（pphm単位）",
        "INDUS": "非小売業の土地面積の割合（人口単位）",
        "RAD": "幹線道路へのアクセス指数",
        "ZN": "25,000平方フィート以上の住宅区画の割合",
        "CHAS": "チャールズ川沿いかどうか（1:Yes、0:No）" 
}