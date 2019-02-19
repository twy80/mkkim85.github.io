---
title: "윈도우에서 TensorFlow 설치"
date: "2019-02-19 11:20:00 +0900"
tags:
  - TensorFlow
  - Anaconda
  - Python
use_math: true
---

## Anaconda 설치
[다운로드 페이지](https://www.anaconda.com/distribution/)에서 아래 이미지처럼 윈도우 버튼을 클릭하고 Python 3.x 버전을 다운로드하고 설치

<figure>
<img src="/assets/images/2019-02-19-install-tensorflow/anaconda.png">
</figure>

## Anaconda 및 Python 패키지 업데이트
<figure>
<img src="/assets/images/2019-02-19-install-tensorflow/anaconda-prompt.png" style="width:392px">
</figure>

Anaconda Prompt 실행 후 아래 명령을 차례대로 입력하여 패키지 업데이트

```
conda update -n  base Anaconda
conda update --installation
```

## TensorFlow 설치
계속해서 아래 명령어를 입력해서 TensorFlow 설치
```
conda install tensorflow
```

TensorFlow 설치 후 테스트를 위해서 Jupyter Notebook 실행 후 `New` -> `Python 3`

<figure>
<img src="/assets/images/2019-02-19-install-tensorflow/jupyter-notebook.png" style="width:392px">
</figure>

{%
include figure
image_path="/assets/images/2019-02-19-install-tensorflow/tensorflow-test1.png"
%}

TensorFlow 테스트를 위해서 아래 코드 입력 후 실행

```
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

TensorFlow 설치가 정상적으로 설치되었다면 아래와 같은 실행 결과가 출력

{%
include figure
image_path="/assets/images/2019-02-19-install-tensorflow/tensorflow-test2.png"
%}

## Refenreces
- [Anaconda](https://www.anaconda.com/)
- [TensorFlow](https://www.tensorflow.org/)
