
# 텐서플로 라이브러리 임포트
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# MNIST 데이터셋 로드, 샘플 값을 정수 -> 부동소수
mnist = tf.keras.datasets.mnist                             #mnist 로드

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0           #계산의 편의를 위해 실수로

# layer를 쌓아 모델 구축. optimizer, loss function 선택
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),            # ReLu
  tf.keras.layers.Dropout(0.2),            	      	    # Dropout
  tf.keras.layers.Dense(10, activation='softmax')	    # softmax
])

# 모델 훈련 및 평가
model.compile(optimizer='adam',				    # adam optimizer
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)