import os, io, base64

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename

import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
import pprint

application = Flask(__name__)
application.config['SECRET_KEY'] = os.urandom(24)

# MNIST Beginner推論
def predict_beginner(srcimg_nparray):
    # 定数定義
    MODEL_DIR ="./model_beginner/"
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
        ximage = srcimg_nparray.flatten().astype(np.float32) / 255.0 #形式を変更
        ximage = np.expand_dims(ximage, axis=0) # (784, 1) ⇒ (1, 784)に変換
        predict = sess.run(y, feed_dict={x: ximage})
        print(predict[0])
        #print('結果：' + str(sess.run(tf.argmax(predict, 1))))
        return(sess.run(tf.argmax(predict, 1)))


@application.route('/', methods=['GET'])
def index():
    return render_template('index.html')

###
# WebAPI送受信サンプル
#
@application.route('/api/predict/beginner', methods=['POST'])
def apitest():
    #application.logger.warn('test message')
    if request.method == 'POST':
        json_data = request.get_json()
        encoded_img = json_data['image']
        decode_data = base64.b64decode( encoded_img.split(',')[1] )

        # バイトストリーム
        img_binarystream = io.BytesIO(decode_data)
        # PILイメージ <- バイナリーストリーム
        img_pil = Image.open(img_binarystream)
        # numpy配列(RGBA?) <- PILイメージ
        img_numpy = np.asarray(img_pil)

        # RGBAの分離（たぶん）
        r, g, b, a = cv2.split(img_numpy)
        # なぜかAのところに白黒反転した絵が保持られているので、それをそのまま入力画像として流用する
        # 細かい理屈はあとで考えよう．．．
        inputimg = cv2.resize(a, (28,28))

        predict = predict_beginner(inputimg)
        print(predict[0])

        result = { 'predict_beginner': str(predict[0]) }
        return jsonify( result )

if __name__ == '__main__':
    application.debug = True
    application.run()
