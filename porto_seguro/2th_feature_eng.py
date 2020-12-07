# porto_predict 2등 코드
(kaggle) porto_seguro_safe_driver_pediction
- Little Boat -
 


1) 중요한 변수 간의 상호 작용 변수(예:ps_cat_13, ps_ind_03, ps_reg_03,...)

2) 범주형 변수의 개수

3) XGBoost 모델의 예측값 : 변수를 3개의 그룹(car, ind, reg)으로 분리해, 2개의 그룹을 변수로 사용하여  3번째 변수 예측할 수 있도록
   XGBoost 모델의 결과값을 변수로 사용한다.

4) 변수 통합 : 두 변수를 선정(예:ps_car_13, ps_ind_03), 하나를 그룹 변수로 사용하고 다른 하나를 값 벼수로 사용해 
              평균, 표준편차, 최댓값, 최솟값, 중간값 등을 계산하여 변수 중요도가 높게 나온 일부 변수만 사용
              
5) 모든 범주형 변수에 임베딩 계층 사용
  (임베딩 계층 차원은 4, dropout은 0.25)

6) NN모델은 2계층으로, ReLU 함수와 높은 값의 dropout(512차원 + 0.8 droupout, 64차원 + 0.8 dropout)을 사용

7) 일부 범주형 변수에 대한 카운트 기반 파생 변수

####
* 트리 모델 학습에는 LightGBM을, 인공 신경망 모델 학습에는 케라스를 사용한다.
* 파일 별 설명

- 피처 엔지니어링
fea_eng0.py
 인공 신경망모델에서 사용할 파생 변수 생성

- 인공 신경망 모델 학습 및 결과 예측
nn_model290.py
- LightGBM 트리 모델 학습 및 결과 예측
dgm_model291.py
=> 각 모델에 알맞는 추가 파생 변수 생성, 내부 교차 검증 후 테스트 데이터에 대한 예측 결과물 생성

- 가중 평균 앙상블
simple_average.py
 두 결과물의 가중 평균 앙상블을 구한다.

[참조] 

*LightGBM 기반 최고 성능 모델
https://www.kaggle.com/xiaozhouwang/2nd-place-lightgbm-solution
*NN기반 최고 성능 모델
https://www.kaggle.com/xiaozhouwang/2nd-place-solution-nn-model
*모든 관련 코드
https://github.com/xiaozhouwang/kaggle-porto-seguro


## 코드 및 데이터 준비

git clone https:/github.com/xiaozhouwang/kaggle-porto-seguro.git 
cd kaggle-porto-seguro/code

## 피처 엔지니어링
- 익명화된 57개의 변수 이름 패턴 : ("ps_[대분류]_[번호]_[데이터 유형]])을 가지고 있다.
- 대분류 카테고리 : ind, car, clac, reg
(추정)
ind - 개인
car - 자동차
calc - calculated
reg - regression

bin - 이진 변수
cat - 범주형 변수

데이터 안에 존재하는 기존 변수를 그 외 변수로 학습해, 기본 변수에 대한 예측 값을 얻은 후, 파생 변수로 활용.
ex) 'car', 'ind', 'reg' 각 대분류 카테고리에 대해 나머지 두 개의 대분류 변수를 기반으로 운전자별 변수 값을 예측한다.


# 다음 코드는 'car,ind,reg'세 가지 대분류에 대한 파생 변수를 생성하는 코드이다.
# 4-13 XGBoost 모델을 통해 데이터의 설명 변수르 다른 설명 변수로 학습한 후, 모델의 계측값을 파생 변수 후보로 저장한다.

# 라이브러리를 불러온다.
import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# XGBoost 모델 설정값 지정
eta = 0.1
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 55
num_boost_round = 500

params = {
    'objective' : 'reg:linear',
    'booster' : 'gbtree',
    'eta' : eta,
    'max_depth' : int(max_depth),   # 트리 모델의 높이를 제한한다.(일반적으로 5-10)
    'subsample' : subsample,        # 행 기준 랜덤 추출
    'colsample_bytree' : colsample_bytree, # 열 기준 랜덤 추출
    'min_child_weight' : min_child_weight,
    'silent':1
    }
# 훈련 데이터, 테스트 데이터 불러와 하나로 통합

train = pd.read_csv('./train.csv')
train_label = train['target']
train_id = train['id']
del train['target'],train['id']

test = pd.read_csv('./test.csv')
test_id = test['id']
del test['id']

data = train.append(test)
data.reset_index(inplace=True)
tarin_rows = train.shape[0]

# 파생 변수 생성
feature = []

for target_g in ['car','ind','reg']:
    # target_g는 예측 대상(target_list)로 사용하고, 그 외 대분류 학습 변수(feature)로 사용한다.
    features=[x for x in list(data) if target_g not in x]
    target_list = [x for x in list(data) if target_g in x]
    train_fea = np.array(data[features])
    for target in target_list:
        print(target)
        train_lael = data[target]
        # 데이터를 5개로 분리하여, 모든 데이터에 대하 예측값을 게산.
        kfold = KFold(n_splits = 5, random_state=218, shuffle=True)
        kf = kfold.split(data)
        cv_train = np.zeros(shape=(data.shape[0], 1))
        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train,label_validate = \
                train_fea[train_fold,:], train_fea[validate,:],train_label[train_fold], train_label[validate]
            dtrain = xgb.DMatrix(X_train, label_train)
            dvalid = xgb.DMatrix(X_validate, label_validate)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            # XGBoost 모델을 학습한다.
            bst = xgb.train(params, dtrain,num_boost_round, svlas = watchlist, verbose_eval=50, early_stopping_rounds=10)
            # 예측 결과물을 저장한다.
            cv_train[validate,0] += bst.predict(xgb.DMatrix(X_validate), ntree_limit=bst.best_ntree_limit)
        feature_results.append(cv_train)
        
# 예측 결과물을 훈련, 테스트 데이터로 분리한 후, pickle로 저장한다.
feature_results = np.hstack(feature_results)
train_feafures = feature_results[:train_rows,:]
test_features = feature_results[train_rows:,:]

import pickle
pickle.dump([train_features, test_features], open('./fea0.pk','wb'))

