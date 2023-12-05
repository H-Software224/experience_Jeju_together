# Experience_Jeju_together

## 프로젝트 계기
안녕하세요. 저희는 이번 KHUDA 4기 Let's experience Jeju together 프로젝트를 하게 된 KHUDA 4기 김민석, 최호윤, 한주상이라고 합니다.
저희는 Dacon에서 제주도 특산물에 대한 가격 예측 경진 대회에 참여했습니다. 구성원 모두 데이터 관련 경진 대회 경험을 얻고 싶어 이 프로젝트를 11.8 ~ 12.3일 까지 진행 하게 되었습니다.

## 회의 계획 내용(feat. 화이트보드)
![plan6](https://github.com/H-Software224/experience_Jeju_together/assets/66872113/bd7673b7-57e2-4b8f-a4b9-c978c0492727)

![plan7](https://github.com/H-Software224/experience_Jeju_together/assets/66872113/7bab799c-7680-4f72-a31a-9cf06c844ca1)

## 데이터 분석 과정(EDA)
1. 날짜 정보 EDA

연도별 가격 평균을 통해 해가 갈수록 가격상승을 파악

<img width="423" alt="plan1" src="https://github.com/H-Software224/experience_Jeju_together/assets/66872113/eaf841cc-2313-4117-a079-3f4537114e9b">

월별 가격 평균을 통해 계절 별로 가격의 변화를 파악

<img width="421" alt="plan2" src="https://github.com/H-Software224/experience_Jeju_together/assets/66872113/1b50bd7e-563d-4aab-83b3-07d9dc9e6a29">

2. 시계열 정보 EDA

<img width="455" alt="plan3" src="https://github.com/H-Software224/experience_Jeju_together/assets/66872113/2d02b7e9-7d0c-4748-a1ef-14c62a5cbe77">

3. 이상치 EDA

<img width="422" alt="plan4" src="https://github.com/H-Software224/experience_Jeju_together/assets/66872113/a2de784a-4933-4486-a1b7-8bb574476720">

## 모델 선정 과정

1. 자동 회귀 예측 통합 이동 평균 분석 모델(ARIMA)로 예측 시도

파이썬 모듈 statsmodels.tsa에서 ARIMA 모델 사용

p = 자기회귀 부분의 차수 d = 1차 차분이 포함된 정도 q = 이동 평균 부분의 차수

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

<img width="454" alt="plan6" src="https://github.com/H-Software224/experience_Jeju_together/assets/66872113/f6c8568c-a408-499a-a6ac-7c054ff9a1db">

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

이에 대한 데이터 시각화 (일정적으로 규칙적으로 변화가 있다는 것을 확인했다.)

<img width="504" alt="plan5" src="https://github.com/H-Software224/experience_Jeju_together/assets/66872113/ef175fea-cc3b-42fe-a2a5-754be7c3932f">

## EDA를 반영한 후처리

1. 공휴일의 데이터 값을 0으로 변경

2. 일요일의 데이터 값을 0으로 변경

3. 음수로 예측한 값을 0으로 변경


## 프로젝트를 통해서 느낀점

1. Dacon 대회에서 제일 중요한 건 점수다.

2. 데이터를 이해하는 것이 제일 중요.

3. 어떤 모델을 쓰냐?(X) 파라미터 튜닝?(X) 전처리 및 후처리를 어떻게 하냐?(O)

4. DACON 대회본선 순위권에 들고 나서 기법에 대해 이해해도 된다.

코드는 github에 파일 올려놨으니 참고해주시기 바랍니다! 

## Reference(참고 문헌)

ARIMA 모듈 URL

<https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>

AutoGluon URL

TimeSeriesDataFrame

<https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html>

TimeSeriesPredictor

<https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>

PPT 템플릿

<https://www.slidemembers.com/en_US/search/ALL/jejudo/1/>

EDA 코드 공유(DACON 참고)

<https://dacon.io/competitions/official/236176/codeshare/9167?page=3&dtype=recent>
