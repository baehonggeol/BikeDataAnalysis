# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:37:34 2020

@author: kogi

이 모델은 두시간 전의 대여소 변동 패턴을 사용하여 예측을 하는 모델이다.
1개 대여소를 먼저 실험 해보았다.
"""

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


bike = pd.read_csv('bikedata.csv')
# dropping nan from data
bike = bike.dropna()

# data type 변환
bike['stack'] = bike['stack'].astype('category')
date_format = '%y%m%d %H%M%S'
bike['time'] = pd.to_datetime(bike['time'], format=date_format)

# dropping testing sites and outliers
indexNames = bike[(bike['stack'] == '1084. 윤선생빌딩(JYP사옥)') | (bike['stack'] == '1309. 보문3교 옆') |
                  (bike['stack'] == '99998. 상암단말정비') | (bike['stack'] == '9996. 시설2')
                  | (bike['stack'] == '1687. 서울월드컵경기장 테스트')
                  | (bike['stack'] == '132. 창천문화공원')
                  | (bike['stack'] == '위트콤') | (bike['stack'] == '위트콤공장')].index
bike.drop(indexNames, inplace=True)


stationlist = bike['stack'].unique()
# 데이터가 원하는 대로 나오는지 체크하기
uni_data = bike[bike["stack"] == stationlist[0]]
uni_data1 = uni_data["Cbike"]
uni_data1.index = uni_data["time"]
uni_data1.plot(subplots=True)

# just getting the data at this point Cbike 값만 빼기
uni_data1 = uni_data1.values

TRAIN_SPLIT = 10000
tf.random.set_seed(13)  # 초기 무게값이 0이면 계속 0이기때문에 랜덤으로로 초기화

# normalizing
uni_train_mean = uni_data1[:TRAIN_SPLIT].mean()
uni_train_std = uni_data1[:TRAIN_SPLIT].std()

# standizing
uni_data1 = (uni_data1 - uni_train_mean) / uni_train_std

univariate_past_history = 12 #한 데이터당 시간은 10분이며 약 두시간을 기준으로 그다음 타겟을 찾는 방식이다.
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data1, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data1, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

print('Single window of past history')
print(x_train_uni[0])
print('\n Target bike to predict')
print(y_train_uni[0])

# 타임 스텝을 - 으로 만든이유는 그만큼 뒤 데이터를 가지고 예측을 할 것이다 라고 말하는 것이다.
def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 50

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

val_univariate.take(1)

for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
