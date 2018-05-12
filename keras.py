# -*- coding: UTF-8 -*-

"""
NNによるボストンハウスのデータ(13種類の指標と住宅価格のデータ）の予測
NNのフレームワークとしてはkerasとtensorflowを使用。

----------------------------------------------------------------------------
<http://liaoyuan.hatenablog.jp/entry/2018/02/03/004849>を参考にしています。


"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

# python 3.6.4 win32
# windows 10 64bit
# keras (2.1.3)
# tensorflow (1.4.0)
# numpy (1.14.0)
# matplotlib (2.1.1)

# モデル定義
def build_model(input_dimx=13, unitx=100):
    model = Sequential()
    model.add(Dense(units=unitx, activation='relu', input_dim=input_dimx ))
    model.add(Dense(units=unitx, activation='relu'))
    model.add(Dense(units=unitx, activation='relu'))
    model.add(Dense(units=unitx, activation='relu'))
    model.add(Dense(units=1))
    # 損失(loss)は2乗mse(mean square error)だが、評価(metrics)は差の絶対値mae(mean absolute error)でおこなっている
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def get_dataset():
    # ボストンハウスのデータのダウンロードまたは読み込み。 
    # 訓練と試験用に分割する。　分割の割合はGBR.pyと同じにしてあります
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data(test_split=0.1, seed=13)
    # データを　平均0、分散1に変換する（正規化）
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std
    return (train_data, train_targets), (test_data, test_targets)

def plot_history(history):
	# 学習時の損失と評価mae(mean absolute error)をプロットする
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['mean_absolute_error'], label='mae')
    plt.title('loss/mae')
    plt.xlabel('epoch')
    plt.ylabel('loss/mae')
    plt.yscale('log') 
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boston Housing: keras')
    parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500, help='Number of epoch')
    parser.add_argument('--unit', '-u', type=int, default=100, help='Number of unit')
    parser.add_argument('--verbose', '-v', type=int, default=1, help='model.predict verbose option')
    args = parser.parse_args()
    
    # データセットの準備
    (train_data, train_targets), (test_data, test_targets)=get_dataset()
    print("dataset is ready.")
    
    # モデル構築とフィッティング
    model = build_model(input_dimx=train_data.shape[1], unitx=args.unit)
    history = model.fit(train_data, train_targets, epochs=args.epoch, batch_size=args.batchsize, verbose=args.verbose)
    
    # テストデータを使って予測誤差を計算する
    # 住宅価格の予想値と実際の価格差の評価のため 誤差の絶対値の平均値MAE(mean absolute error)をつかう
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print("MSE(mean square error): %.4f" % test_mse_score)
    print("MAE(mean absolute error): %.4f" % test_mae_score)
    
    # 予測値の計算
    Y_predict= model.predict(test_data, batch_size= args.batchsize, verbose=0, steps=None)
    Y_predict= Y_predict.reshape(len(Y_predict))
    
    print("")
    # 予測誤差が大きいWorst 10を表示する
    index0= np.argsort( np.abs(test_targets - Y_predict))[::-1]
    print ('Worst 10: estimation, actual, difference')
    for i in index0[0:10]:
        print(test_targets[i], "%.4f" % Y_predict[i], "%.4f" % (test_targets[i]- Y_predict[i]) )
    
    # 学習時の損失と評価mae(mean absolute error)をepoch毎にプロットする
    plot_history(history)
    