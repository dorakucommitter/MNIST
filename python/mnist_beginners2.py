#!/usr/bin/env python
# coding: utf-8

# # ライブラリのインポート
# 共通機能
import time
# tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 開始時間のタイムスタンプ取得
start_time = time.time()
print('開始時刻: %f' % start_time)


# # MNIST画像データの読込み
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")


# # 訓練画像、正解ラベル等のパラメータ定義

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# モデル
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交差エントロピー
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 勾配降下法を用い交差エントロピーが最小となるようyを最適化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()

# Sessionを開始する
with tf.Session() as sess:
    sess.run(init)
    
    print("--- 訓練開始 ---")
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    print("--- 訓練終了 ---")


    # 認識精度の確認
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
    # 精度の計算
    # correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
    # Trueならば1、Falseならば0に変換される
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # 精度の実行と表示
    # テストデータの画像とラベルで精度を確認する
    # ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
    print("精度")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


    # 終了時刻
    end_time = time.time()
    print("終了時刻: " + str(end_time))
    print("かかった時間: " + str(end_time - start_time))
    
    
    ############################################################################
    ### このスクリプトが獲得した成果物の表示
    
    ### バイアスb配列
    bias = sess.run(b)
    print(bias)

    ### 重みW
    # 画像認識結果の描画
    import matplotlib.cm as cm
    from matplotlib import pylab as plt
    
    # Weightの値
    weights = sess.run(W)
    
    fig, axarr = plt.subplots(2, 5)
    for idx in range(10):
        ax = axarr[int(idx / 5)][idx % 5]
        ax.imshow(weights[:, idx].reshape(28, 28), cmap = cm.Greys_r)
        ax.set_title(str(idx))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.close()

