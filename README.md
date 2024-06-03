# BME-CAPSTONE 

Title : UPDRS Prediction Using Time Series Data of Protein/Peptide <br/>
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

## Train Model
_compare_model.py_

For regression, we tried XGBoost, SVM, Random Forest model and the ensemble of those models. Model training was conducted using the pre-processed data which we just did as input, and each MSE was obtained and the results were compared.

![스크린샷 2024-06-04 013043](https://github.com/kse27/BME-CAPSTONE/assets/145419092/32a8b504-c981-4651-ac59-b329a899d642)


## Extract Important Feature
_feature_extraction_50.py_

모델 학습 결과 MSE 가 가장 작게 나온 XGB(XGBoost)+RF(RandomForest) 모델을 선택하여 이 모델 학습 과정 중에 중요하게 학습되는 feature 50개를 추출하였습니다. 이 과정에서 각 UPDRS 별로 다른 feature들을 변수로 저장하여, 이후에 LSTM 모델 학습에 사용하였습니다.

## Train LSTM model


## Conclusion


## Reference
DATA - https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data <br/>
Project 
(updrs, regression model, LSTM model)
