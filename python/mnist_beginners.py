#!/usr/bin/env python
# coding: utf-8

# # MNISTチュートリアルサイト
# * [TensorFlow MNIST For ML Beginners](https://www.tensorflow.org/tutorials)
# * [TensorFlowチュートリアル - ML初心者のためのMNIST（翻訳） - Qiita](http://qiita.com/KojiOhki/items/ff6ae04d6cf02f1b6edf)
# * [初心者必読！MNIST実行環境の準備から手書き文字識別までを徹底解説！ | 技術ブログ | MIYABI Lab](https://miyabi-lab.space/blog/10)
# 
# ## 考え方
# * [TensorFlowコトハジメ 手書き文字認識(MNIST)による多クラス識別問題](http://yaju3d.hatenablog.jp/entry/2016/04/22/073249)

# # ライブラリのインポート

# In[ ]:


# 共通機能
import time
# tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# # 開始時間のタイムスタンプ取得

# In[ ]:


# 開始時刻
start_time = time.time()
print('開始時刻: %f' % start_time)


# # MNIST画像データの読込み
# * 訓練用画像 55000件 (60000件という説もあるらしい)
# * テスト用画像 10000件
# * 訓練データとテストデータは、それぞれ0～9の画像とそれぞれの画像に対応するラベル（0～9）が設定されている
# * 画像サイズは28px X 28px(=784)
# * mnist.train.imagesは[55000, 784]の配列
# * mnist.train.lablesは、read_data_setsメソッドのone_hotのT/Fにより下記の通り  
#     * one_hot = Trueの場合 : [55000, 10]の配列で、対応するimagesの画像が「3」の時、[0,0,0,1,0,0,0,0,0,0]
#     ```
#     np.argmax([0,0,0,1,0,0,0,0,0,0]) ⇒ 3に変換できる
#     ```
#     * one_hot = Falseの場合: [55000, 1]の配列で、対応するimagesの画像が「3」の時、3  
# * mnist.test.imagesは[10000, 784]の配列、mnist.test.lablesは[10000, 10]の配列で、内容はmnist.trainと同様
# * mnist.validation.imagesというデータも5,000件ほど存在するが、チュートリアルでは登場しない。

# In[ ]:


print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")


# # 訓練画像、正解ラベル等のパラメータ定義

# In[ ]:


# 訓練画像を入れる変数
# 訓練画像は28x28pxであり、これらを1行784列の画素配列に並び替えて格納する
# Noneとなっているのは、訓練画像をいくつ入れるか変数定義時点ではわからないため、
# いくつでも格納できるようにするための手法
# C言語でいうところのNULL Arrayみたいな感じ？
x = tf.placeholder(tf.float32, [None, 784])

# 正解データのラベル
# 0～9までの数値は、数値ではなく10個の配列が代入される
# ラベルの値=7の場合：  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
y_ = tf.placeholder(tf.float32, [None, 10])

# 重み
# 訓練画像のpx数の行、ラベル（0-9の数字の個数）数の列の行列
# 初期値として0を入れておく
W = tf.Variable(tf.zeros([784, 10]))

# バイアス
# ラベル数の列の行列
# 初期値として0を入れておく
b = tf.Variable(tf.zeros([10]))


# # モデルの作成
# 
# ## 大雑把な考え方
# mnist_beginnerでは、画素配列の１要素ごとに１次関数で0～9らしさのようなものを算出し、全体に占める割合が一番大きかったラベルを識別結果と判断します。なので、数式的には、
# 
#     y = x * W + b
#       W: 重み(weight)
#       b: バイアス
# 
# を画素の数だけ計算します。
# とりあえず、重みWで数字らしさを評価し、バイアスで数値ごとの総和を調整する、のように考えると、なんとなく納得できるような気がします。
# 
# ### バイアスを足す理由（推測）
# グレースケールで黒⇒白の色調を0⇒255の数値で表すため、認識結果が、白が多い数値ほど`x * W`が大きく、黒が多いほど`x * W`の値が小さくなり、数値そのものより画像全体に占める白黒の割合に左右されてしまいます（たぶん）。このため、バイアスを使って総和の調整をしている、と勝手に思っています。
# 
# ## ソフトマックス、交差エントロピー、勾配降下法
# * [損失関数）クロスエントロピー誤差](https://qiita.com/celaeno42/items/7efdbb1491406f4bde96)
# * [TensorFlowのクロスエントロピー関数の動作](https://qiita.com/exy81/items/8c9ab6ba8d4a03873d7c)
# * [ソフトマックス関数とシグモイド関数でのクロスエントロピー誤差関数について](http://lapislazuli.home.localnet:8888/notebooks/ipynb/mnist_beginners.ipynb)
# 
# * 勾配降下法
#     * [勾配降下法ってなんだろう - 白猫のメモ帳](https://shironeko.hateblo.jp/entry/2016/10/29/173634)
#     * [2017-08-27
# ディープラーニング(深層学習)を理解してみる(勾配降下法：最急降下法と確率的勾配降下法)](http://yaju3d.hatenablog.jp/entry/2017/08/27/233459)
#     * [確率的勾配降下法とは何か、をPythonで動かして解説する](http://lapislazuli.home.localnet:8888/notebooks/ipynb/mnist_beginners.ipynb)

# In[ ]:


# ソフトマックス
# yは入力x（画像）に対し、それがある数字である確率の配列
# matmul関数で行列xとWの掛け算を行った後、bを加算する。
# yは[1, 10]の行列
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交差エントロピー
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 勾配降下法を用い交差エントロピーが最小となるようyを最適化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# In[ ]:


# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()

# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）

sess = tf.Session()
sess.run(init)

# 1000回の訓練（train_step）を実行する
# next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
# feed_dictでplaceholderに値を入力することができる
print("--- 訓練開始 ---")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
print("--- 訓練終了 ---")


# # 認識精度の確認

# In[ ]:


# テスト画像がどの数字であるかの予測yと正解ラベルy_を比較する
# 同じ値であればTrueが返される
# argmaxは配列の中で一番値の大きい箇所のindexが返される
# 一番値が大きいindexということは、それがその数字である確率が一番大きいということ
# Trueが返ってくるということは訓練した結果と回答が同じということ
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


# In[ ]:


# 終了時刻
end_time = time.time()
print("終了時刻: " + str(end_time))
print("かかった時間: " + str(end_time - start_time))


# ---
# # このスクリプトが獲得した成果物の表示

# ## バイアスb配列

# In[ ]:


bias = sess.run(b)
print(bias)


# ## 重みW

# In[ ]:


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

