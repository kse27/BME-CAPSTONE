# -*- coding: utf-8 -*-
"""BME Capstone

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1c1vqEGlqlw-oNHchHQOYDC3zXh1Z6Gmw

# 🧸 BME Capstone
:predicting MDS-UPDR scores by using protein and peptide levels

👉 202100805 김시은, 202102238 유희진

"""# **PCA**

**UPDRS1: MSE=14**
"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# 데이터 불러오기
updrs1 = pd.read_csv("/content/drive/MyDrive/BME/updrs1.csv")

# 각 환자의 고유한 patient_id 확인
unique_patients = updrs1['patient_id'].unique()

# 각 환자를 train 및 test 세트로 분할
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

# train 및 test 세트에 속하는 인덱스 추출
train_idx = updrs1['patient_id'].isin(train_patients)
test_idx = updrs1['patient_id'].isin(test_patients)

# train 및 test 데이터 분할
train_data = updrs1[train_idx]
test_data = updrs1[test_idx]

# Extract patient IDs
train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

# Drop non-feature columns and the target column
train_features = train_data.drop(columns=['patient_id','visit_id','visit_month','updrs_1']).columns
test_features = test_data.drop(columns=['patient_id','visit_id','visit_month','updrs_1']).columns

# Extract target columns
train_targets = train_data['updrs_1']
test_targets = test_data['updrs_1']

train_patient_sequences = []
test_patient_sequences = []
train_target_sequences = []
test_target_sequences = []

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][train_features].values
    patient_target = train_data[train_data['patient_id'] == patient_id]['updrs_1'].values
    train_patient_sequences.append(patient_data)
    train_target_sequences.append(patient_target)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][test_features].values
    patient_target = test_data[test_data['patient_id'] == patient_id]['updrs_1'].values
    test_patient_sequences.append(patient_data)
    test_target_sequences.append(patient_target)

# 최대 시퀀스 길이 계산 (중간 차원의 최대 길이)
max_seq_len = max(len(patient) for patient in train_patient_sequences)
max_seq_len_t = max(len(patient) for patient in test_patient_sequences)

print(max_seq_len)
print(max_seq_len_t)

# 각 시퀀스를 동일한 길이로 패딩
data_3d_padded = pad_sequences(train_patient_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testdata_3d_padded = pad_sequences(test_patient_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 타겟 시퀀스를 동일한 길이로 패딩
target_3d_padded = pad_sequences(train_target_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testtarget_3d_padded = pad_sequences(test_target_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 3차원 배열로 변환
data_3d_padded = np.array(data_3d_padded)
testdata_3d_padded = np.array(testdata_3d_padded)
target_3d_padded = np.array(target_3d_padded)
testtarget_3d_padded = np.array(testtarget_3d_padded)

print("3D array shape:", data_3d_padded.shape)
print("3D array shape:", testdata_3d_padded.shape)
print("Target 3D array shape:", target_3d_padded.shape)
print("Test Target 3D array shape:", testtarget_3d_padded.shape)

# 3차원 데이터를 2차원으로 변환 (198*max_seq_len, 1196)
num_samples, num_timesteps, num_features = data_3d_padded.shape
num_samplest, num_timestepst, num_featurest = testdata_3d_padded.shape

data_2d = data_3d_padded.reshape(num_samples * num_timesteps, num_features)
testdata_2d = testdata_3d_padded.reshape(num_samplest * num_timestepst, num_featurest)

# 데이터 표준화
scaler = StandardScaler()
data_2d_scaled = scaler.fit_transform(data_2d)
testdata_2d_scaled = scaler.fit_transform(testdata_2d)

# PCA 모델 생성
pca = PCA(n_components=100)

# PCA 모델을 사용하여 데이터를 2차원으로 압축
data_2d_compressed = pca.fit_transform(data_2d_scaled)
testdata_2d_compressed = pca.fit_transform(testdata_2d_scaled)

# 압축된 데이터를 다시 3차원으로 변환 (198, max_seq_len, 100)
pca_3d_sequences = data_2d_compressed.reshape(num_samples, num_timesteps, 100)
tpca_3d_sequences = testdata_2d_compressed.reshape(num_samplest, num_timestepst, 100)

# 타겟 컬럼 추가 (198, max_seq_len, 101)
pca_3d_sequences_with_target = np.concatenate([pca_3d_sequences, target_3d_padded[..., np.newaxis]], axis=-1)
tpca_3d_sequences_with_target = np.concatenate([tpca_3d_sequences, testtarget_3d_padded[..., np.newaxis]], axis=-1)

print("PCA 3D sequences with target shape:", pca_3d_sequences_with_target.shape)
print("Test PCA 3D sequences with target shape:", tpca_3d_sequences_with_target.shape)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, 101), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in pca_3d_sequences_with_target:
    X_train = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_train = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_train).any():
        continue

    X_train = np.expand_dims(X_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_train = np.expand_dims(y_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    model.fit(X_train, y_train, epochs=50, batch_size=16)

# 모델 예측
predictions = []
true_values = []

for sequence in tpca_3d_sequences_with_target:
    X_test = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_test = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_test).any():
        continue

    X_test = np.expand_dims(X_test, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_pred = model.predict(X_test)
    predictions.append(y_pred.flatten())

    # 현재 시퀀스의 마지막 실제값이 NaN이 아니면 true_values에 추가합니다.
    last_value = sequence[-1][-1]
    if not np.isnan(last_value):
        true_values.append(last_value)

# Flatten the predictions list and true_values list
predicted_values = np.concatenate(predictions)
true_values = np.array(true_values)

# Evaluation metrics
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

"""**UPDRS2 : MSE=20**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# 데이터 불러오기
updrs2 = pd.read_csv("/content/drive/MyDrive/BME/updrs2.csv")

# 각 환자의 고유한 patient_id 확인
unique_patients = updrs2['patient_id'].unique()

# 각 환자를 train 및 test 세트로 분할
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

# train 및 test 세트에 속하는 인덱스 추출
train_idx = updrs2['patient_id'].isin(train_patients)
test_idx = updrs2['patient_id'].isin(test_patients)

# train 및 test 데이터 분할
train_data = updrs2[train_idx]
test_data = updrs2[test_idx]

# Extract patient IDs
train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

# Drop non-feature columns and the target column
train_features = train_data.drop(columns=['patient_id','visit_id','visit_month','updrs_2']).columns
test_features = test_data.drop(columns=['patient_id','visit_id','visit_month','updrs_2']).columns

# Extract target columns
train_targets = train_data['updrs_2']
test_targets = test_data['updrs_2']

train_patient_sequences = []
test_patient_sequences = []
train_target_sequences = []
test_target_sequences = []

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][train_features].values
    patient_target = train_data[train_data['patient_id'] == patient_id]['updrs_2'].values
    train_patient_sequences.append(patient_data)
    train_target_sequences.append(patient_target)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][test_features].values
    patient_target = test_data[test_data['patient_id'] == patient_id]['updrs_2'].values
    test_patient_sequences.append(patient_data)
    test_target_sequences.append(patient_target)

# 최대 시퀀스 길이 계산 (중간 차원의 최대 길이)
max_seq_len = max(len(patient) for patient in train_patient_sequences)
max_seq_len_t = max(len(patient) for patient in test_patient_sequences)

print(max_seq_len)
print(max_seq_len_t)

# 각 시퀀스를 동일한 길이로 패딩
data_3d_padded = pad_sequences(train_patient_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testdata_3d_padded = pad_sequences(test_patient_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 타겟 시퀀스를 동일한 길이로 패딩
target_3d_padded = pad_sequences(train_target_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testtarget_3d_padded = pad_sequences(test_target_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 3차원 배열로 변환
data_3d_padded = np.array(data_3d_padded)
testdata_3d_padded = np.array(testdata_3d_padded)
target_3d_padded = np.array(target_3d_padded)
testtarget_3d_padded = np.array(testtarget_3d_padded)

print("3D array shape:", data_3d_padded.shape)
print("3D array shape:", testdata_3d_padded.shape)
print("Target 3D array shape:", target_3d_padded.shape)
print("Test Target 3D array shape:", testtarget_3d_padded.shape)

# 3차원 데이터를 2차원으로 변환 (198*max_seq_len, 1196)
num_samples, num_timesteps, num_features = data_3d_padded.shape
num_samplest, num_timestepst, num_featurest = testdata_3d_padded.shape

data_2d = data_3d_padded.reshape(num_samples * num_timesteps, num_features)
testdata_2d = testdata_3d_padded.reshape(num_samplest * num_timestepst, num_featurest)

# 데이터 표준화
scaler = StandardScaler()
data_2d_scaled = scaler.fit_transform(data_2d)
testdata_2d_scaled = scaler.fit_transform(testdata_2d)

# PCA 모델 생성
pca = PCA(n_components=100)

# PCA 모델을 사용하여 데이터를 2차원으로 압축
data_2d_compressed = pca.fit_transform(data_2d_scaled)
testdata_2d_compressed = pca.fit_transform(testdata_2d_scaled)

# 압축된 데이터를 다시 3차원으로 변환 (198, max_seq_len, 100)
pca_3d_sequences = data_2d_compressed.reshape(num_samples, num_timesteps, 100)
tpca_3d_sequences = testdata_2d_compressed.reshape(num_samplest, num_timestepst, 100)

# 타겟 컬럼 추가 (198, max_seq_len, 101)
pca_3d_sequences_with_target = np.concatenate([pca_3d_sequences, target_3d_padded[..., np.newaxis]], axis=-1)
tpca_3d_sequences_with_target = np.concatenate([tpca_3d_sequences, testtarget_3d_padded[..., np.newaxis]], axis=-1)

print("PCA 3D sequences with target shape:", pca_3d_sequences_with_target.shape)
print("Test PCA 3D sequences with target shape:", tpca_3d_sequences_with_target.shape)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, 101), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in pca_3d_sequences_with_target:
    X_train = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_train = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_train).any():
        continue

    X_train = np.expand_dims(X_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_train = np.expand_dims(y_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    model.fit(X_train, y_train, epochs=50, batch_size=16)

# 모델 예측
predictions = []
true_values = []

for sequence in tpca_3d_sequences_with_target:
    X_test = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_test = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_test).any():
        continue

    X_test = np.expand_dims(X_test, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_pred = model.predict(X_test)
    predictions.append(y_pred.flatten())

    # 현재 시퀀스의 마지막 실제값이 NaN이 아니면 true_values에 추가합니다.
    last_value = sequence[-1][-1]
    if not np.isnan(last_value):
        true_values.append(last_value)

# Flatten the predictions list and true_values list
predicted_values = np.concatenate(predictions)
true_values = np.array(true_values)

# Evaluation metrics
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

"""**UPDRS3: MSE=143**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# 데이터 불러오기
updrs3 = pd.read_csv("/content/drive/MyDrive/BME/updrs3.csv")

# 각 환자의 고유한 patient_id 확인
unique_patients = updrs3['patient_id'].unique()

# 각 환자를 train 및 test 세트로 분할
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

# train 및 test 세트에 속하는 인덱스 추출
train_idx = updrs3['patient_id'].isin(train_patients)
test_idx = updrs3['patient_id'].isin(test_patients)

# train 및 test 데이터 분할
train_data = updrs3[train_idx]
test_data = updrs3[test_idx]

# Extract patient IDs
train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

# Drop non-feature columns and the target column
train_features = train_data.drop(columns=['patient_id','visit_id','visit_month','updrs_3']).columns
test_features = test_data.drop(columns=['patient_id','visit_id','visit_month','updrs_3']).columns

# Extract target columns
train_targets = train_data['updrs_3']
test_targets = test_data['updrs_3']

train_patient_sequences = []
test_patient_sequences = []
train_target_sequences = []
test_target_sequences = []

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][train_features].values
    patient_target = train_data[train_data['patient_id'] == patient_id]['updrs_3'].values
    train_patient_sequences.append(patient_data)
    train_target_sequences.append(patient_target)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][test_features].values
    patient_target = test_data[test_data['patient_id'] == patient_id]['updrs_3'].values
    test_patient_sequences.append(patient_data)
    test_target_sequences.append(patient_target)

# 최대 시퀀스 길이 계산 (중간 차원의 최대 길이)
max_seq_len = max(len(patient) for patient in train_patient_sequences)
max_seq_len_t = max(len(patient) for patient in test_patient_sequences)

print(max_seq_len)
print(max_seq_len_t)

# 각 시퀀스를 동일한 길이로 패딩
data_3d_padded = pad_sequences(train_patient_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testdata_3d_padded = pad_sequences(test_patient_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 타겟 시퀀스를 동일한 길이로 패딩
target_3d_padded = pad_sequences(train_target_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testtarget_3d_padded = pad_sequences(test_target_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 3차원 배열로 변환
data_3d_padded = np.array(data_3d_padded)
testdata_3d_padded = np.array(testdata_3d_padded)
target_3d_padded = np.array(target_3d_padded)
testtarget_3d_padded = np.array(testtarget_3d_padded)

print("3D array shape:", data_3d_padded.shape)
print("3D array shape:", testdata_3d_padded.shape)
print("Target 3D array shape:", target_3d_padded.shape)
print("Test Target 3D array shape:", testtarget_3d_padded.shape)

# 3차원 데이터를 2차원으로 변환 (198*max_seq_len, 1196)
num_samples, num_timesteps, num_features = data_3d_padded.shape
num_samplest, num_timestepst, num_featurest = testdata_3d_padded.shape

data_2d = data_3d_padded.reshape(num_samples * num_timesteps, num_features)
testdata_2d = testdata_3d_padded.reshape(num_samplest * num_timestepst, num_featurest)

# 데이터 표준화
scaler = StandardScaler()
data_2d_scaled = scaler.fit_transform(data_2d)
testdata_2d_scaled = scaler.fit_transform(testdata_2d)

# PCA 모델 생성
pca = PCA(n_components=100)

# PCA 모델을 사용하여 데이터를 2차원으로 압축
data_2d_compressed = pca.fit_transform(data_2d_scaled)
testdata_2d_compressed = pca.fit_transform(testdata_2d_scaled)

# 압축된 데이터를 다시 3차원으로 변환 (198, max_seq_len, 100)
pca_3d_sequences = data_2d_compressed.reshape(num_samples, num_timesteps, 100)
tpca_3d_sequences = testdata_2d_compressed.reshape(num_samplest, num_timestepst, 100)

# 타겟 컬럼 추가 (198, max_seq_len, 101)
pca_3d_sequences_with_target = np.concatenate([pca_3d_sequences, target_3d_padded[..., np.newaxis]], axis=-1)
tpca_3d_sequences_with_target = np.concatenate([tpca_3d_sequences, testtarget_3d_padded[..., np.newaxis]], axis=-1)

print("PCA 3D sequences with target shape:", pca_3d_sequences_with_target.shape)
print("Test PCA 3D sequences with target shape:", tpca_3d_sequences_with_target.shape)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, 101), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in pca_3d_sequences_with_target:
    X_train = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_train = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_train).any():
        continue

    X_train = np.expand_dims(X_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_train = np.expand_dims(y_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    model.fit(X_train, y_train, epochs=50, batch_size=16)

# 모델 예측
predictions = []
true_values = []

for sequence in tpca_3d_sequences_with_target:
    X_test = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_test = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_test).any():
        continue

    X_test = np.expand_dims(X_test, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_pred = model.predict(X_test)
    predictions.append(y_pred.flatten())

    # 현재 시퀀스의 마지막 실제값이 NaN이 아니면 true_values에 추가합니다.
    last_value = sequence[-1][-1]
    if not np.isnan(last_value):
        true_values.append(last_value)

# Flatten the predictions list and true_values list
predicted_values = np.concatenate(predictions)
true_values = np.array(true_values)

# Evaluation metrics
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

"""**UPDRS4: MSE=**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# 데이터 불러오기
updrs4 = pd.read_csv("/content/drive/MyDrive/BME/updrs4.csv")

# 각 환자의 고유한 patient_id 확인
unique_patients = updrs4['patient_id'].unique()

# 각 환자를 train 및 test 세트로 분할
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

# train 및 test 세트에 속하는 인덱스 추출
train_idx = updrs4['patient_id'].isin(train_patients)
test_idx = updrs4['patient_id'].isin(test_patients)

# train 및 test 데이터 분할
train_data = updrs4[train_idx]
test_data = updrs4[test_idx]

# Extract patient IDs
train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

# Drop non-feature columns and the target column
train_features = train_data.drop(columns=['patient_id','visit_id','visit_month','updrs_4']).columns
test_features = test_data.drop(columns=['patient_id','visit_id','visit_month','updrs_4']).columns

# Extract target columns
train_targets = train_data['updrs_4']
test_targets = test_data['updrs_4']

train_patient_sequences = []
test_patient_sequences = []
train_target_sequences = []
test_target_sequences = []

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][train_features].values
    patient_target = train_data[train_data['patient_id'] == patient_id]['updrs_4'].values
    train_patient_sequences.append(patient_data)
    train_target_sequences.append(patient_target)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][test_features].values
    patient_target = test_data[test_data['patient_id'] == patient_id]['updrs_4'].values
    test_patient_sequences.append(patient_data)
    test_target_sequences.append(patient_target)

# 최대 시퀀스 길이 계산 (중간 차원의 최대 길이)
max_seq_len = max(len(patient) for patient in train_patient_sequences)
max_seq_len_t = max(len(patient) for patient in test_patient_sequences)

print(max_seq_len)
print(max_seq_len_t)

# 각 시퀀스를 동일한 길이로 패딩
data_3d_padded = pad_sequences(train_patient_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testdata_3d_padded = pad_sequences(test_patient_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 타겟 시퀀스를 동일한 길이로 패딩
target_3d_padded = pad_sequences(train_target_sequences, maxlen=max_seq_len, dtype='float32', padding='post')
testtarget_3d_padded = pad_sequences(test_target_sequences, maxlen=max_seq_len_t, dtype='float32', padding='post')

# 3차원 배열로 변환
data_3d_padded = np.array(data_3d_padded)
testdata_3d_padded = np.array(testdata_3d_padded)
target_3d_padded = np.array(target_3d_padded)
testtarget_3d_padded = np.array(testtarget_3d_padded)

print("3D array shape:", data_3d_padded.shape)
print("3D array shape:", testdata_3d_padded.shape)
print("Target 3D array shape:", target_3d_padded.shape)
print("Test Target 3D array shape:", testtarget_3d_padded.shape)

# 3차원 데이터를 2차원으로 변환 (198*max_seq_len, 1196)
num_samples, num_timesteps, num_features = data_3d_padded.shape
num_samplest, num_timestepst, num_featurest = testdata_3d_padded.shape

data_2d = data_3d_padded.reshape(num_samples * num_timesteps, num_features)
testdata_2d = testdata_3d_padded.reshape(num_samplest * num_timestepst, num_featurest)

# 데이터 표준화
scaler = StandardScaler()
data_2d_scaled = scaler.fit_transform(data_2d)
testdata_2d_scaled = scaler.fit_transform(testdata_2d)

# PCA 모델 생성
pca = PCA(n_components=100)

# PCA 모델을 사용하여 데이터를 2차원으로 압축
data_2d_compressed = pca.fit_transform(data_2d_scaled)
testdata_2d_compressed = pca.fit_transform(testdata_2d_scaled)

# 압축된 데이터를 다시 3차원으로 변환 (198, max_seq_len, 100)
pca_3d_sequences = data_2d_compressed.reshape(num_samples, num_timesteps, 100)
tpca_3d_sequences = testdata_2d_compressed.reshape(num_samplest, num_timestepst, 100)

# 타겟 컬럼 추가 (198, max_seq_len, 101)
pca_3d_sequences_with_target = np.concatenate([pca_3d_sequences, target_3d_padded[..., np.newaxis]], axis=-1)
tpca_3d_sequences_with_target = np.concatenate([tpca_3d_sequences, testtarget_3d_padded[..., np.newaxis]], axis=-1)

print("PCA 3D sequences with target shape:", pca_3d_sequences_with_target.shape)
print("Test PCA 3D sequences with target shape:", tpca_3d_sequences_with_target.shape)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, 101), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in pca_3d_sequences_with_target:
    X_train = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_train = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_train).any():
        continue

    X_train = np.expand_dims(X_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_train = np.expand_dims(y_train, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    model.fit(X_train, y_train, epochs=50, batch_size=16)

# 모델 예측
predictions = []
true_values = []

for sequence in tpca_3d_sequences_with_target:
    X_test = sequence[:-1, :]  # t일 때의 데이터를 입력으로 사용
    y_test = sequence[1:, -1]  # t+1일 때의 updrs 점수를 타겟으로 사용

    # null 값을 가진 행을 스킵합니다.
    if np.isnan(y_test).any():
        continue

    X_test = np.expand_dims(X_test, axis=0)  # LSTM 모델 입력을 위해 차원 확장
    y_pred = model.predict(X_test)
    predictions.append(y_pred.flatten())

    # 현재 시퀀스의 마지막 실제값이 NaN이 아니면 true_values에 추가합니다.
    last_value = sequence[-1][-1]
    if not np.isnan(last_value):
        true_values.append(last_value)

# Flatten the predictions list and true_values list
predicted_values = np.concatenate(predictions)
true_values = np.array(true_values)

# Evaluation metrics
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
