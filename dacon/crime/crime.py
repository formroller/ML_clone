# import os
# os.getcwd()
# os.chdir('C:\/Users/BIOJEAN/kaggle/kaggle_crime')

import pandas as pd
import pandas_profiling
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
pd.options.display.max_columns=100

train = pd.read_csv('./train.csv', parse_dates=['Dates'])
test = pd.read_csv('./test.csv',parse_dates=['Dates'], index_col='Id')

# 데이터 확인
# profile = train.profile_report()

train.head()
test.head()

train.info()
test.info()

# 결측치 없음
train.isnull().sum()
test.isnull().sum()

# 파생변수 생성
def feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Dates'].dt.date)
    data['n_days'] = (data['Date'] - data['Date'].min()).apply(lambda x : x.days)
    data['Day'] = data['Dates'].dt.day
    data['DayOfWeek'] = data['Dates'].dt.weekday
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Block'] = data['Address'].str.contains('block', case=False).apply(lambda x : 1 if x== True else 0)
    data['X_Y'] = data['X'] - data['Y']
    data['XY'] = data['X']+data['Y']
    data.drop(columns = ['Dates','Date','Address'], inplace=True)
    return data

train = feature_engineering(train)
test = feature_engineering(test)
train.drop(columns=['Descript','Resolution'], inplace = True)


train.head()
test.head()

# 경찰서 이름 범주화
le1 = LabelEncoder()
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
test['PdDistrict'] = le1.fit_transform(test['PdDistrict'])

# 훈련 데이터 생성
le2 = LabelEncoder()
X = train.drop(columns=['Category'])
y = le2.fit_transform(train['Category'])

train.head()
X.head()

train_data = lgb.Dataset(X, label=y, categorical_feature=['PdDistrict', ])
params = {'boosting':'gbdt',
          'objective':'multiclass',
          'num_class':39,
          'max_delta_setp':0.9,
          'min_data_in_leaf':21,
          'learning_rate':0.4,
          'max_bin':465,
          'num_leaves':41,
          'verbose':1
          }


bst = lgb.train(params, train_data, 120)
predictions=bst.predict(test)


submission = pd.DataFrame(predictions, columns=le2.inverse_transform(np.linspace(0,38,39,dtype='int16')), index=test.index)
submission.to_csv('LGBM_classification.csv',index_label='Id')


train_data = lgb.Dataset(X, label=y, categorical_feature=['PdDistrict', ])
params = {'boosting':'gbdt',
          'objective':'multiclass',
          'num_class':39,
          'max_delta_step':0.9,
          'min_data_in_leaf': 21,
          'learning_rate': 0.4,
          'max_bin': 465,
          'num_leaves': 41,
          'verbose' : 1}

bst = lgb.train(params, train_data, 120)
predictions = bst.predict(test)


# 참고
1.kaggle) San Francisco Crime Classification, https://www.kaggle.com/junheo/sf-crime-rate-prediction
2. Lightgbm, https://www.kaggle.com/junheo/sf-crime-rate-prediction

* Light GBM : 트리 기반 학습 알고리즘인 gradient boosting 방식의 프레임 워크이다.
* Ligth GBM(이하 lgbm)은 나무를 수직으로 확장한다. 따라서 leaf-wise tree gorwth인 LGBM은 최대 delta loss가 증가하도록 잎의 개수를 정한다.
  leaf-wise 알고리즘은 다름 level-wise(수평적 확장) 알고리즘보다 낮은 loss를 달성하는 경향이 있다.
  단, 데이터의 크기가 작은 경우 leaf-wise는 과적합(overfitting)되기 쉬우므로 max_depth를 줄여야 한다.

(주요 파라미터)
* objective : regression, binary, multiclass..
* metric : mae, rmse, mape, binary_logloss, auc, corss_entropy, kullbac_leibler..
* boosting : (default=gbdt), gbdt, rf, dart, goss

1) learning_rate : 일반적으로 0.01~0.1로 맞추고 다른 파라미터를 튜닝한다. 나중에 성능을 더 높일 경우 learning_rate를 줄인다.
2) num_iterations : 기본값은 100이나 1000정도가 적당.(너무 클 경우 과적합 발생)
3) max_depth : -1로 설정하면 제한없기 분기한다. feature가 많다면 크게 설정한다. 파라미터 설정 시 우선적으로 설정.
4) boosting : (부스팅 방법) 기본값은 gbdt이며 정확도가 중요할때는 딥러닝의 드랍아웃 같은 dart를 사용한다. 샘플링을 이용하는 goss도 있다.
5) bagginf_fraction : 배깅을 하기위해 데이터를 랜덤 샘플링해 학습에 사용한다.
                     (비율은 0<fraction<1이며 0이 되지 않게 해야한다.)    
6) feature_fraction : 1보다 작다면 LGBM은 매 iteration(tree)마다 다름 feature를 랜덤하게 추출해 학습하게 된다. 만약, 0.8로 값을 설정하면 매 tree 구성시
                     feature의 80%만 랜덤하게 선택한다. 과적합을 방지하기 위해 사용할 수 있으며 학습 속도가 향상된다.
7) scale_pos_weight : 클래스 불균형의 데이터 셋에서 weigth를 주는 방식으로 positive를 증가시킨다.
                     기본값으 1이며, 불균형의 정도에 따라 조절한다.
8) early_stopping_round : Validation 셋에서 평가지표가 더 이상 향상되지 않으면 학습을 정지한다. 평가지표의 향상이 n round 이상 지속되면 학습을 정지한다.
9) lambda_l1, lambda_l2 : 정규화를 통해 과적합을 방지할 수 있지만, 정화도를 저하시킬수 있기 때문에 일반적으로 default 값인 0으로 둔다.

[빠른 속도]
 - bagging_fraction
 - max_bin은 작게
 - save_binary 사용시 데이터 로딩 속도 향상
 - parallel learning 사용

[높은 정확도]
 - max_bin 크게
 - num_iterations 크게하고, learning_rate는 작게
 - num_leaves를 크게 (과적합의 원인이 될 수 있다.)
 - boosting 알고리즘 'dart' 사용
 
[과적합 줄이기]
 - max_bin 삭제
 - num_leaves 작게
 - min_data_in_leaf와 min_sum_hessian_in_leaf 사용하기.
