# BME-CAPSTONE 

Title : **UPDRS Prediction Using Time Series Data of Protein/Peptide** <br/>
Member : BME 21 김시은, BME 21 유희진 <br/>
Date : 2024.03 - 2024.06 <br/>


## Data Preprocessing
_data_preprocessing.py_

1. Merge 3 Raw Data
- #1 Grouping
- #2 Pivoting
- #3 Merging

2. NULL 값 처리
* 같은 환자에 대해서 펩타이드/단백질 정보를 가지고 있으나 특정 월에 존재하지 않을 때, 각 환자 별 평균값으로 NULL 값을 대체하였습니다
* 같은 환자에 대해서 특정 펩타이드/단백질 정보를 가지고 있지 않다면 환자에게 해당 정보가 없다고 판단하여 0으로 대체하였습니다

<img width="800" alt="KakaoTalk_20240604_161941782" src="https://github.com/kse27/BME-CAPSTONE/assets/145419092/3db404af-11c4-4db4-a754-7b05e4179db0">

## Overview
<img width="800" alt="KakaoTalk_20240612_201143021" src="https://github.com/kse27/BME-CAPSTONE/assets/145419092/03625cb3-77b1-4f85-8c66-6fd0a8365e2b">


## Train Model
_compare_model.py_

회귀문제 해결을 위해, XGBoost, SVM, Random Forest 모델과 각각을 앙상블하여 만든 모델들을 사용하여 결과를 비교해보았습니다. 우리가 방금 한 전처리 데이터를 입력으로 사용하여 모델 학습을 진행하였으며, 각 MSE를 구하여 모델간의 결과를 비교했습니다. 그결과 XGBoost 와 Random Forest를 합친 모델이 4가지의 지표에 있어서 모두 좋은 결과가 나타났음을 알 수 있었습니다. 

![image](https://github.com/kse27/BME-CAPSTONE/assets/145419092/12753828-81bd-47b1-9292-6cfea930c8ea)


## Extract Important Feature
_feature_extraction_50.py_

모델 학습 결과 MSE 가 가장 작게 나온 XGB(XGBoost)+RF(RandomForest) 모델을 선택하여 이 모델 학습 과정 중에 중요하게 학습되는 feature 50개를 추출하였습니다. 이 과정에서 각 UPDRS 별로 다른 feature들을 변수로 저장하여, 이후에 LSTM 모델 학습에 사용하였습니다.

![스크린샷 2024-06-04 013420](https://github.com/kse27/BME-CAPSTONE/assets/145419092/df5bf007-a145-4240-a0ff-d82517cd254e)

## Train LSTM model
_lstm.py_
_pca.py_

Feature Selection 과 PCA를 이용하여 LSTM 모델 학습을 진행하였습니다. 모델의 layer의 unit 수는 50으로 하여 진행하였으며 dropout 을 통해 과적합을 방지하였습니다.

## Result
![image](https://github.com/kse27/BME-CAPSTONE/assets/145419092/83a0453f-9f75-474e-b2fc-afb9fe1fac38)

## Discussion
1. Model이 중요하게 학습한  feature 분석
파킨슨병 연구에서 중요하게 보는 단백질(ex. Alpha-synuclein)과 파킨슨병과의 연관성이 아직 알려지지 않은 단백질(ex. Tau Protein)이 UPDRS 점수를 예측할 때 중요한 feature로 작용했다는 결과
 현재는 파킨슨 병과 직접적인 연관성이 밝혀지지 않은 단백질 혹은 펩타이드도 향후 연구를 통해 해당 질병과의 상관관계를 밝힐 수 있을 거라 기대

   
2. Feature Selection vs. PCA (Principal Component Analysis)
PCA 기법을 이용하여 LSTM 모델을 학습하여 예측을 진행했던 것이 feature importance 기반으로 뽑아서 학습한 것보다 더 좋은 성능을 나타냄
PCA 가 데이터의 본질적 구조를 더 보존하고 많은 정보를 포함하여 차원 축소를 수행하였기 때문이라고 예상
성능 향상을 위해, auto encoder 와 같은 다른 차원 축소 기법을 이용해 볼 필요 있음

   
## Reference
DATA - https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data <br/>
