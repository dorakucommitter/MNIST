import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename

import base64, io
import cv2
import numpy as np
from PIL import Image

import tensorflow as tf


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
TMPIMG = UPLOAD_FOLDER + '/tmp.png'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def index():
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    return render_template('index.html')

# MNIST Beginner推論
def predict_beginner(srcimg_nparray):
    # 定数定義
    MODEL_DIR ="./model/"
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

@app.route('/send', methods=['POST'])
def send():
    if request.method == 'POST':
        data = request.form.get('img')
        decode_data = base64.b64decode( data.split(',')[1] )
        #with open(TMPIMG, 'wb') as f:
        #    f.write(decode_data)
        #    f.close()

        # バイトストリーム
        img_binarystream = io.BytesIO(base64.b64decode( data.split(',')[1] ))
        # PILイメージ <- バイナリーストリーム
        img_pil = Image.open(img_binarystream)
        # numpy配列(RGBA?) <- PILイメージ
        img_numpy = np.asarray(img_pil)

        # RGBAの分離（たぶん）
        r, g, b, a = cv2.split(img_numpy)
        # なぜかAのところに白黒反転した絵が保持られているので、それをそのまま入力画像として流用する
        # 細かい理屈はあとで考えよう．．．
        inputimg = cv2.resize(a, (28,28))
        #print(inputimg)
        #cv2.imwrite(UPLOAD_FOLDER + '/tmp2.png', inputimg)

        predict = predict_beginner(inputimg)
        print(predict)

        return render_template('index.html', predict=100)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True
    app.run()
