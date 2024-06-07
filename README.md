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

-데이터에 시계열 데이터를 포함하여 LSTM 모델으로 학습해보았습니다. 이를 통해, 각 환자마다 달마다 변화하는 추이를 학습하여 다음 방문때의 UPDRS 를 예측할 수 있도록 학습을 진행하였으며, 실제로 이전 각 case를 independent하게 진행하였을때보다 조금더 정확한 결과가 나왔음을 확인할 수 있었습니다. 

## Overview
<img width="800" alt="KakaoTalk_20240604_161933410" src="https://github.com/kse27/BME-CAPSTONE/assets/145419092/86d70500-521f-4d4e-9dd0-296e989c0037">

## Conclusion
1. UPDRS 점수를 예측하는 것에 중요하게 사용된 feature를 알아냄으로써, 각 updrs 점수당 어떤 단백질/펩타이드가 중요한 영향을 끼치는지를 알아볼 수 있었습니다. 이를 통해, 실제로 아직 파킨슨병과의 연관성이 알려지지 않은 단백질이지만 추후에 연관성이 알려지기를 기대할 수 있었습니다.
   
2. 시계열 데이터를 활용함으로써, 개인의 단백질 차이에 의해 다르게 예측되는 UPDRS 점수를 예측할 수 있었습니다. 이를 통해, 추후에 이 모델을 활용하게 된다면, 주어진 단백질 정보를 통해 예상되는 UPDRS 점수를 예측하여 의학적 분야에 사용될 수 있을거라 기대합니다.
   
## Reference
DATA - https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data <br/>
