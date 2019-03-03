#!/usr/bin/env python
# coding: utf-8

# # ライブラリのインポート
# 共通機能
import os
import shutil
import time
# tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定数定義
MNIST_DATA_DIR = "../MNIST_data/"
TENSORBOARD_DIR = "./tfboard/"
MODEL_DIR ="./model/"
RESULT_FILE = "%sresult.ckpt" % MODEL_DIR

# 開始時間のタイムスタンプ取得
start_time = time.time()
print('開始時刻: %f' % start_time)


# # MNIST画像データの読込み
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets(MNIST_DATA_DIR, one_hot=True)
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
tf.summary.scalar("cross_entropy", cross_entropy)

# 勾配降下法を用い交差エントロピーが最小となるようyを最適化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

# 作成済みモデルが存在する場合は、一旦削除して再作成
# 学習⇒中断⇒学習再開で精度を出すのは、結構難しいらしい。
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    os.mkdir(MODEL_DIR)
else:
    os.mkdir(MODEL_DIR)

# Sessionを開始する
with tf.Session() as sess:
    # 学習結果の保存
    saver = tf.train.Saver()

    # 変数の初期化の実行？
    sess.run(tf.global_variables_initializer())

    # TensorBoardで表示する値の設定
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TENSORBOARD_DIR, graph_def=sess.graph_def)

    print("--- 訓練開始 ---")
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
        if step % 10 == 0:
            train_accur = sess.run(accuracy,feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d, training accuracy %g" % (step,train_accur))
        # 1 step終わるたびにTensorBoardに表示する値を追加する
        summary_str = sess.run(summary_op, feed_dict={x: batch_xs, y_: batch_ys})
        summary_writer.add_summary(summary_str, step)
    print("--- 訓練終了 ---")

    # 学習結果の保存
    saver.save(sess, RESULT_FILE)

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
