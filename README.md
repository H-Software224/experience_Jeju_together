# experience_Jeju_together

## 주제를 정한 동기
안녕하세요. 저희는 이번 KHUDA 4기 Let's experience Jeju together 프로젝트를 하게 된 KHUDA 4기 김민석, 최호윤 그리고 한주상이라고 합니다.
Dacon에서 우연히 제주도에서 특산물에 대해서 가격을 예측하는 것이 행사를 하는 것을 보게 되어서, 이번 KHUDA 동아리에서 정규 프로젝트인 만큼 데이터를 분석하고 싶어서 이거 아니면 데이터 분석 예측을 더이상 할 기회가 없다고 생각하였습니다. 그래서 이 주제로 프로젝트를 11.8 ~ 12.3일 까지 하게 되었습니다.

## 데이터 분석 과정(EDA) 이용
1. 날짜 정보 EDA

연도별 가격 평균을 통해 해가 갈수록 가격상승을 파악

2. 시계열 정보 EDA

3. 이상치 EDA
## 모델 선정 과정

1. 자동 회귀 예측 통합 이동 평균 분석 모델(ARIMA)로 예측 시도

파이썬 모듈 statsmodels.tsa에서 ARIMA 모델 사용

p = 자기회귀 부분의 차수
d = 1차 차분이 포함된 정도
q = 이동 평균 부분의 차수

```python
# 모델 학습하고 예측하기
ar_400 = TSA.arima.model.ARIMA(train_element.loc[:, 'total_price(원)'], order=(3, 1, 3))
ar_400_mod = ar_400.fit()
predict_total_price = ar_400_mod.predict("2023-03-04", "2023-03-31", dynamic=True)
ar_400_supply = TSA.arima.model.ARIMA(train_tg_A_J.loc[:, 'supply(kg)'], order=(3, 1, 3))
ar_400_supply_mod = ar_400_supply.fit()
predict_supply = ar_400_supply_mod.predict("2023-03-04", "2023-03-31", dynamic=True)
ar_400 = ar_400.initialize()
ar_400_supply = ar_400_supply.initialize()
test_price_list+= (predict_total_price / predict_supply).to_list()
```
이를 통한 ACF 그래프로 시각화을 하였습니다.




2. 자동 머신러닝 학습 모듈(AutoGulon) 시도

```python
# 모델을 위한 데이터준비
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
tr_time_df = TimeSeriesDataFrame(train_df.drop(columns=['ID','corporation','location','item']))
test_time_df = TimeSeriesDataFrame(test_df.drop(columns=['ID','corporation','location', 'item']))
model = TimeSeriesPredictor(prediction_length = 28,target = 'price(원/kg)', known_covariates_names=["year", "month", "day", "weekday"], eval_metric='RMSE')
model.fit(tr_time_df,presets="best_quality", time_limit=20000, random_seed=42) # fitting
result = model.predict(tr_time_df, known_covariates=test_time_df, random_seed=42) # 예측(테스트 데이터를 이용)
```

이에 대한 데이터 시각화 (일정적으로 규칙적으로 변화가 있다는 것을 알 수 있다.)

## 마지막으로 전처리 때 썼던 EDA 방법을 이용하여 후처리

1. 공휴일의 데이터 값을 0으로 변경
2. 일요일의 데이터 값을 0으로 변경
3. 음수로 예측한 값을 0으로 변경


## 프로젝트를 통해서 알 수 있는 점

1. DACON에서 나온 경진대회 이외에 대회에서 제일 중요한 건 점수다(중요한 건 꺾이지 않는 마음으로 점수에 임한다.)
2. 데이터를 이해하는 것이 제일 중요
3. 어떤 모델을 쓰냐? 파라미터 튜닝(X) 전처리 및 후처리를 어떻게 하냐?(O)
4. 앞으로 DACON 대회본선에 들고 나서 기법에 대해 알아도 된다.

코드는 github에 파일 올려놨으니 참고해주시기 바랍니다! 감사합니다!




