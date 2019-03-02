#!/usr/bin/env python
# coding: utf-8

# # 各種ライブラリのインポート

# In[ ]:


# tensorflowライブラリのインポート
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 画像配列表示
import matplotlib.cm as cm
from matplotlib import pylab as plt

import numpy as np


# # スクリプト内定数設定

# In[ ]:



# MNIST画像データ保存場所
DATADIR = '../MNIST_data/'

# ラベル形式設定
# one_hot == True: 画像が「7」の場合、label=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# one_hoe != True: 画像が「7」の場合、label=7
is_one_hot = True


# # MNIST画像の読み込み

# In[ ]:


# 画像の読み込み
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets(DATADIR, one_hot=is_one_hot)
print("--- MNISTデータの読み込み完了 ---")


# # MNIST画像データ数の確認
# * 訓練画像   : 55,000
# * テスト画像 : 10,000
# * 検証画像   : 5,000
# 「テスト」と「検証」の差がわからない．．．

# In[ ]:


# 文字列の出力方法は3パターンある

### 訓練画像 ###
print('mnist.train.images = ' + str(len(mnist.train.images)))
### 検証画像 ###
print('mnist.test.images = {:d}'.format(len(mnist.test.images)))
### 検証画像? ###
print('mnist.validation.images = %d' % len(mnist.validation.images))


# # 訓練画像の先頭画像表示

# In[ ]:


# 画像情報

# データラベル形式
if is_one_hot == True:
    label = np.argmax(mnist.train.labels[0])
else:
    label = mnist.train.labels[0]

plt.imshow(mnist.train.images[0].reshape(28, 28), cmap = cm.Greys_r)
plt.title(str(label))
plt.axis('off')
plt.show()
plt.close()


# In[ ]:


# 画素配列
print(np.round(mnist.train.images[0].reshape(28, 28)).astype(np.int64))


# # 訓練画像、テスト画像、検証画像の先頭10枚の表示

# In[ ]:


# 訓練画像
# 2行x5列の画像出力領域を確保
fig, axarr = plt.subplots(2, 5)
# 各出力領域に絵をセットする
for idx in range(10):
    ax = axarr[int(idx / 5)][idx % 5]
    ax.imshow(mnist.train.images[idx].reshape(28, 28), cmap = cm.Greys_r)

    label = ''
    if is_one_hot == True:
        label = np.argmax(mnist.train.labels[idx])
    else:
        label = mnist.train.labels[idx]
    ax.set_title(str(label))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
# 絵を出力する
plt.show()
plt.close()


# In[ ]:


# テスト画像
# 2行x5列の画像出力領域を確保
fig, axarr = plt.subplots(2, 5)
# 各出力領域に絵をセットする
for idx in range(10):
    ax = axarr[int(idx / 5)][idx % 5]
    ax.imshow(mnist.test.images[idx].reshape(28, 28), cmap = cm.Greys_r)

    label = ''
    if is_one_hot == True:
        label = np.argmax(mnist.test.labels[idx])
    else:
        label = mnist.test.labels[idx]
    ax.set_title(str(label))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
# 絵を出力する
plt.show()
plt.close()


# In[ ]:


# 検証画像
# 2行x5列の画像出力領域を確保
fig, axarr = plt.subplots(2, 5)
# 各出力領域に絵をセットする
for idx in range(10):
    ax = axarr[int(idx / 5)][idx % 5]
    ax.imshow(mnist.validation.images[idx].reshape(28, 28), cmap = cm.Greys_r)

    label = ''
    if is_one_hot == True:
        label = np.argmax(mnist.validation.labels[idx])
    else:
        label = mnist.validation.labels[idx]
    ax.set_title(str(label))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
# 絵を出力する
plt.show()
plt.close()

