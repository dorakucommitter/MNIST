#!/usr/bin/env python
# coding: utf-8

# # ライブラリのインポート
# 共通機能
import os
import shutil
import time
# tensorflow
import tensorflow as tf
# OpenCV
import cv2

# 認識結果の描画
import numpy as np
import matplotlib.cm as cm
from matplotlib import pylab as plt

# 定数定義
MODEL_DIR ="./model/"
#RESULT_FILE = "%sresult.ckpt" % MODEL_DIR

# Model Function
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# モデル
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Sessionを開始する
with tf.Session() as sess:
    # 学習済みモデルのロード
    if os.path.exists(MODEL_DIR):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        saver.restore(sess, ckpt.model_checkpoint_path) # 変数データの読み込み

    # 画像読み込み
    img = input("画像ファイルのパスを入力してください >")
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    ximage = img.flatten().astype(np.float32)/255.0 #形式を変更
    ximage = np.expand_dims(ximage, axis=0) # (784, 1) ⇒ (1, 784)に変換
    predict = sess.run(y, feed_dict={x: ximage})
    print(predict[0])
    print('結果：' + str(sess.run(tf.argmax(predict, 1))))
    #print("入力画像" + str(sess.run(y, feed_dict={x: ximage})))
    plt.imshow(ximage.reshape(28, 28), cmap = cm.Greys_r)
    plt.title('input picture')
    plt.axis('off')
    plt.show()
    plt.close()
