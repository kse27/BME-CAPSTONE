# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
data = pd.read_csv("/content/drive/MyDrive/BME/modified_data2.csv")  # 데이터 파일 경로에 맞게 수정해주세요.

"""환자를 8:2로 나눔"""

from sklearn.model_selection import train_test_split

# 각 환자의 고유한 patient_id 확인
unique_patients = data['patient_id'].unique()

# 각 환자를 train 및 test 세트로 분할
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

# train 및 test 세트에 속하는 인덱스 추출
train_idx = data['patient_id'].isin(train_patients)
test_idx = data['patient_id'].isin(test_patients)


# train 및 test 데이터 분할
train_data = data[train_idx]
test_data = data[test_idx]

#print(train_data,test_data)

train_data.isnull().any()


"""# **UPDRS1_TOP 50**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd

# 각 환자의 데이터를 따로 처리하여 시계열 형태로 만들기
train_patient_sequences = []
test_patient_sequences = []

# 'updrs_1'을 포함한 데이터셋에서 필요한 피처만 선택
features = ['FIYGGC(UniMod_4)GGNR', 'P36980', 'NVVYTC(UniMod_4)NEGYSLIGNPVAR', 'AGLAASLAGPHSIVGR', 'KYLYEIAR',
            'FNKPFVFLM(UniMod_35)IEQNTK', 'Q9NYU2', 'LLELTGPK', 'C(UniMod_4)FSGQC(UniMod_4)ISK', 'GLGEISAASEFK',
            'VVEESELAR', 'P00748', 'GKRPYQEGTPC(UniMod_4)SQC(UniMod_4)PSGYHC(UniMod_4)K', 'LVYPSC(UniMod_4)EEK',
            'LEEQAQQIR', 'DALSSVQESQVAQQAR', 'LEPGQQEEYYR', 'TATSEYQTFFNPR', 'HYTNPSQDVTVPC(UniMod_4)PVPPPPPC(UniMod_4)C(UniMod_4)HPR',
            'VGGVQSLGGTGALR', 'LGPLVEQGR', 'C(UniMod_4)MC(UniMod_4)PAENPGC(UniMod_4)R', 'SGIEC(UniMod_4)QLWR',
            'GC(UniMod_4)PTEEGC(UniMod_4)GER', 'M(UniMod_35)ADEAGSEADHEGTHSTKR', 'SIVVSPILIPENQR', 'MKYWGVASFLQK',
            'FFLC(UniMod_4)QVAGDAK', 'P06310', 'GNSYFMVEVK', 'GATLALTQVTPQDER', 'EILSVDC(UniMod_4)STNNPSQAK',
            'P17174', 'NILDRQDPPSVVVTSHQAPGEK', 'GLSAEPGWQAK', 'LLPAQLPAEKEVGPPLPQEAVPLQK', 'KTLLSNLEEAKK',
            'ADSGEGDFLAEGGGVR', 'FTILDSQGK', 'C(UniMod_4)LVEKGDVAFVKHQTVPQNTGGK', 'SC(UniMod_4)VGETTESTQC(UniMod_4)EDEELEHLR',
            'LETPDFQLFK', 'ILGPLSYSK', 'P20774', 'KVESELIKPINPR', 'GAAPPKQEFLDIEDP', 'P16070', 'SVPMVPPGIK',
            'DKETC(UniMod_4)FAEEGKK', 'P27169', 'updrs_1']

# train_data = pd.read_csv('/content/drive/MyDrive/BME/modified_data2.csv')
# test_data = pd.read_csv('/path/to/test_data.csv')  # 테스트 데이터 경로 설정

train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][features].values
    train_patient_sequences.append(patient_data)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][features].values
    test_patient_sequences.append(patient_data)

# 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, len(features) - 1), return_sequences=True))  # 각 시퀀스의 길이를 None으로 설정
model.add(Dropout(0.2))  # 드롭아웃 레이어를 추가하여 과적합을 방지합니다.
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.001)  # 원하는 학습률로 설정합니다.
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in train_patient_sequences:
    X_train = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

for sequence in test_patient_sequences:
    X_test = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

# MSE 계산
predicted_values = np.concatenate(predictions)
mse = np.mean(np.square(np.array(true_values) - predicted_values))
print("Mean Squared Error:", mse)

"""# **UPDRS2_TOP50**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd

# 각 환자의 데이터를 따로 처리하여 시계열 형태로 만들기
train_patient_sequences = []
test_patient_sequences = []

# 'updrs_2'을 포함한 데이터셋에서 필요한 피처만 선택
features =['FIYGGC(UniMod_4)GGNR', 'LQDLYSIVR', 'MKYWGVASFLQK', 'P30086', 'C(UniMod_4)LPVTAPENGK', 'ETYGEMADC(UniMod_4)C(UniMod_4)AK', 'LEPGQQEEYYR', 'GLSAEPGWQAK', 'KYFIDFVAR',
           'M(UniMod_35)YLGYEYVTAIR', 'EQLSLLDRFTEDAKR', 'GAQTQTEEEMTR', 'P07998', 'NPDSSTTGPWC(UniMod_4)YTTDPTVR', 'SC(UniMod_4)VGETTESTQC(UniMod_4)EDEELEHLR',
           'SC(UniMod_4)DKTHTC(UniMod_4)PPC(UniMod_4)PAPELLGGPSVFLFPPKPK', 'LVNEVTEFAK', 'LSKELQAAQAR', 'P19827', 'ALANSLAC(UniMod_4)QGK', 'TSAHGNVAEGETKPDPDVTER',
           'HYEGSTVPEK', 'VKDISEVVTPR', 'Q9NYU2', 'THLGEALAPLSK', 'P01591', 'EDC(UniMod_4)NELPPRR', 'AGC(UniMod_4)VAESTAVC(UniMod_4)R', 'AGLAASLAGPHSIVGR', 'P07333',
           'EHVAHLLFLR', 'P01034', 'QVVAGLNFR', 'KVESELIKPINPR', 'KTLLSNLEEAK', 'P36980', 'HYTNPSQDVTVPC(UniMod_4)PVPPPPPC(UniMod_4)C(UniMod_4)HPR', 'GWVTDGFSSLK',
           'VLLDGVQNPR', 'HLDSVLQQLQTEVYR', 'QRQEELC(UniMod_4)LAR', 'IGDQWDKQHDMGHMMR', 'YGLVTYATYPK', 'VDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSK', 'NFPPSQDASGDLYTTSSQLTLPATQC(UniMod_4)LAGK',
           'SSGLVSNAPGVQIR', 'LTASAPGYLAITK', 'KQINDYVEKGTQGK', 'ADSGEGDFLAEGGGVR', 'TLKIENVSYQDKGNYR','updrs_2']

train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][features].values
    train_patient_sequences.append(patient_data)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][features].values
    test_patient_sequences.append(patient_data)

# 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, len(features) - 1), return_sequences=True))  # 각 시퀀스의 길이를 None으로 설정
model.add(Dropout(0.2))  # 드롭아웃 레이어를 추가하여 과적합을 방지합니다.
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.001)  # 원하는 학습률로 설정합니다.
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in train_patient_sequences:
    X_train = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

for sequence in test_patient_sequences:
    X_test = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

# MSE 계산
predicted_values = np.concatenate(predictions)
mse = np.mean(np.square(np.array(true_values) - predicted_values))
print("Mean Squared Error:", mse)

"""# **UPDRS3_TOP50**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd

# 각 환자의 데이터를 따로 처리하여 시계열 형태로 만들기
train_patient_sequences = []
test_patient_sequences = []

# 'updrs_3'을 포함한 데이터셋에서 필요한 피처만 선택
features = ['TPC(UniMod_4)TVSC(UniMod_4)NIPVVSGKEC(UniMod_4)EEIIR', 'QHVVYGPWNLPQSSYSHLTR', 'FIYGGC(UniMod_4)GGNR', 'LLEVPEGR', 'P13521', 'GLPAPIEK',
            'ETYGEMADC(UniMod_4)C(UniMod_4)AK', 'Q6UXD5', 'KLGQSLDC(UniMod_4)NAEVYVVPWEK', 'SDVMYTDWKK', 'LADGGATNQGRVEIFYR', 'MYLGYEYVTAIR',
            'LSYTC(UniMod_4)EGGFR', 'KLVGYLDR', 'VTSIQDWVQK', 'QTHQPPAPNSLIR', 'FM(UniMod_35)ETVAEK', 'KDSGFQM(UniMod_35)NQLR', 'P10643', 'YANC(UniMod_4)HLAR',
            'M(UniMod_35)YLGYEYVTAIR', 'HYEGSTVPEK', 'P25311', 'P08123', 'KMTVTDQVNC(UniMod_4)PK', 'NPDSSTTGPWC(UniMod_4)YTTDPTVR', 'GYPGVQAPEDLEWER',
            'NFPPSQDASGDLYTTSSQLTLPATQC(UniMod_4)LAGK', 'GSPAINVAVHVFR', 'C(UniMod_4)PNPPVQENFDVNKYLGR', 'P01877', 'P04406', 'EGDMLTLFDGDGPSAR', 'VFQEPLFYEAPR',
            'RLGMFNIQHC(UniMod_4)K', 'QFTSSTSYNR', 'EKLQDEDLGFL', 'THLGEALAPLSK', 'YQC(UniMod_4)YC(UniMod_4)YGR', 'DPTFIPAPIQAK', 'VTEIWQEVMQR', 'SDVMYTDWK',
            'IEIPSSVQQVPTIIK', 'YPSLSIHGIEGAFDEPGTK', 'THPHFVIPYR', 'IASFSQNC(UniMod_4)DIYPGKDFVQPPTK', 'RLEGQEEEEDNRDSSMK', 'Q13332', 'TLKIENVSYQDKGNYR', 'C(UniMod_4)FSGQC(UniMod_4)ISK','updrs_3']


train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][features].values
    train_patient_sequences.append(patient_data)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][features].values
    test_patient_sequences.append(patient_data)

# 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, len(features) - 1), return_sequences=True))  # 각 시퀀스의 길이를 None으로 설정
model.add(Dropout(0.2))  # 드롭아웃 레이어를 추가하여 과적합을 방지합니다.
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Adam optimizer를 사용하여 모델을 컴파일하고, 학습률을 조정합니다.
adam = Adam(learning_rate=0.001)  # 원하는 학습률로 설정합니다.
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in train_patient_sequences:
    X_train = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

for sequence in test_patient_sequences:
    X_test = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

# MSE 계산
predicted_values = np.concatenate(predictions)
mse = np.mean(np.square(np.array(true_values) - predicted_values))
print("Mean Squared Error:", mse)

"""# **UPDRS4_TOP50**"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd

# 각 환자의 데이터를 따로 처리하여 시계열 형태로 만들기
train_patient_sequences = []
test_patient_sequences = []

# 'updrs_4'을 포함한 데이터셋에서 필요한 피처만 선택
features = ['VNHVTLSQPK', 'SILENLR', 'STGGISVPGPMGPSGPR', 'M(UniMod_35)LTPEHVFIHPGWK', 'DVLLEAC(UniMod_4)C(UniMod_4)ADGHR', 'AQC(UniMod_4)GGGLLGVR',
            'SVPPSASHVAPTETFTYEWTVPK', 'VC(UniMod_4)PFAGILENGAVR', 'HGNVAEGETKPDPDVTER', 'EHVAHLLFLR', 'KLSSWVLLMK', 'LEPGQQEEYYR', 'EKLQDEDLGFL',
            'C(UniMod_4)PFPSRPDNGFVNYPAKPTLYYK', 'LLDNWDSVTSTFSK', 'Q92520', 'TPSAAYLWVGTGASEAEK', 'PPTSAHGNVAEGETKPDPDVTER', 'WSRPQAPITGYR', 'Q99435',
            'SFQTGLFTAAR', 'MMAVAADTLQR', 'SVPMVPPGIK', 'AADDTWEPFASGK', 'P23083', 'Q99683', 'FIYGGC(UniMod_4)GGNR', 'QFTSSTSYNR', 'AYLEEEC(UniMod_4)PATLRK',
            'FVVTDGGITR', 'VPFDAATLHTSTAMAAQHGMDDDGTGQK', 'GC(UniMod_4)PTEEGC(UniMod_4)GER', 'QKVEPLRAELQEGAR', 'RTHLPEVFLSK', 'Q92876', 'AIGAVPLIQGEYMIPC(UniMod_4)EK',
            'VYC(UniMod_4)DMNTENGGWTVIQNR', 'TSAHGNVAEGETKPDPDVTER', 'AGLAASLAGPHSIVGR', 'AGAAAGGPGVSGVC(UniMod_4)VC(UniMod_4)K', 'RLEAGDHPVELLAR', 'EDC(UniMod_4)NELPPRR',
            'GSPSGEVSHPR', 'C(UniMod_4)LAPLEGAR', 'VTGVVLFR', 'GNSYFMVEVK', 'LDEVKEQVAEVR', 'ATEDEGSEQKIPEATNR', 'SVIPSDGPSVAC(UniMod_4)VK', 'C(UniMod_4)VC(UniMod_4)PVSNAMC(UniMod_4)R','updrs_4']


train_patients = train_data['patient_id'].unique()
test_patients = test_data['patient_id'].unique()

for patient_id in train_patients:
    patient_data = train_data[train_data['patient_id'] == patient_id][features].values
    train_patient_sequences.append(patient_data)

for patient_id in test_patients:
    patient_data = test_data[test_data['patient_id'] == patient_id][features].values
    test_patient_sequences.append(patient_data)

# 모델 정의
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, len(features) - 1), return_sequences=True))
model.add(Dropout(0.2))  # 드롭아웃 레이어를 추가하여 과적합을 방지합니다.
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='mean_squared_error')

# 모델 훈련
for sequence in train_patient_sequences:
    X_train = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

for sequence in test_patient_sequences:
    X_test = sequence[:-1, :-1]  # t일 때의 데이터를 입력으로 사용
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

# MSE 계산
predicted_values = np.concatenate(predictions)
mse = np.mean(np.square(np.array(true_values) - predicted_values))
print("Mean Squared Error:", mse)
