# =============================================================================
# 인공 신경망 모델 학습
# =============================================================================
인공 신경망(Neral networt) 기반 모델은 이전 파생변수(porto_feature_eng.py)외 다향한 피처 엔지니어링 과정을 수행한다.

1. 운전자 데이터별 결측값의 개수를 나타내는 missing 변수
2. 변수 조합 간의 곱셈/나눗셈 등의 상호 작용 변수(Interaction Features)
3. 특정 변수 그룹의 값 전체를 하나의 문자영로 통합해 변수 그룹 내 조합을 나타내는 변수
4. 범주형 변수의 빈로를 나타내는 count 변수
5. 5특정 변수(group_column)에 피벗해 타겟 변수(target_column)의 통계값(평균,표준편차,최댓값,최솟값)을 사용하는 피벗 변수

=> 인공 신경망은 -1 ~ 1 사이의 값을 입력값으로 받을 때에 가장 효과적인 학습 결과를 보인다.
모든 피처 엔지니어링을 완료한 후, 변수별 최댓값과 평균값을 기준으로 정규화해 최종 변수를 생성.

# 4-14 인공 신경망 모델 학습을 위해 훈련 데이터와 테스트 데이터를 읽어온다.

# 인공 신경망 모델 keras 라이브러리 읽어오기
from sklearn.layers import Dense, Dropout, Embedding, Fatten, Input, merge
from sklearn.layers.normalizaion import BatchNormalization
from sklearn.layers.advanced_activations import PReLU
from keras.models import Model

# 시간 측정 및 압축 파일을 읽어오기 위한 라이브러리
from time import time
import datetime
from itertools import combinations
import pickle

# 피처 엔지니어링을 위한 라이브러리
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from slkearn.model_selection import StratifiedKFold

# =============================================================================
# =============================================================================
# READ DATA
tarin = pd.read_csv('./train.csv')
train_label = train['target']
train_id = train['id']
del train['target'], tarin['id']

test = pd.read_csv('./test.csv')
test_id = test['id']
del test['id']

# =============================================================================
# =============================================================================
# UTIL FUNCTIONS

# 4-15 파생 변수 생성을 위한 도구 함수를 정의하기.
# interaction_features : 상호 작용 변수를 생성하는 함수. 
# proj_num_on_cat : 피벗 기반 기초 통계 변수 생성하는 함수

def proj_num_on_cat(train_df, test_df, target_column, group_column):
    # train_df : 훈련 데이터
    # test_df : 테스트 데이터
    # target_column : 통계 기반 파생 변수를 생성한 타겟 변수
    # group_column : 피벗(pivot)을 수행할 변수
    train_df['row_id'] = range(train_df.shape[0])
    test_df['row_id'] = range(test_df.shape[0])
    train_df['train'] = 1
    test_df['train'] = 0
    
    # 훈련 데이터와 테스트 데이터를 통합한다.
    all_df = train_df[['row_id','train',target_column,group_column]].append(test_df[['row_id','train',target_column, group_column]])
    
    # group_column 기반을 피벗한 target_column의 값을 구한다.
    grouped = all_df[[target_column, group_column]].groupby(group_column)
    
    # 빈도(size), 평균(mean), 표준편차(std), 중간값(median), 최댓값(max), 최솟값(min)을 구한다.
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size' % target_column]
    the_mean = pd.DataFrame(grouped.mean()).reset_size
    the_mean.columns = [group_column, '%s_mean' % target_column]
    the_std = pd.DataFrame(grouped.std()).reset_index()
    the_std.columns = [group_column, '%s_std' % target_column]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median' % target_column]
    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max' % target_column]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min' % target_column] 
    
    # 통계 기반 파생 변수를 취합한다.
    the_stats = pd.merge(the_size, the_mean)
    the_state = pd.merge(the_state, the_std)
    the_state = pd.merge(the_state, the_median)
    the_state = pd.merge(the_state, the_max)
    the_state = pd.merge(the_state, the_min)
    all_df = pd.merge(all_df, the_state, how = 'left')
    
    # 훈련 데이터와 테스트 데이터로 분리해 반환한다.
    selected_train = all_df[all_df['train'] == 1]
    selected_test = all_df[all_df['train'] == 0]
    selected_train.sort_values('row_id', inplace = True)
    selected_test.sort_values('row_id', inplcae = True)
    selected_train.drop([target_column, group_column, 'row_id','train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'row_id','train'], axis=1, inplace=True)
    selected_train, selected_test = np.asarray(selected_train), np.array(selected_test)
    

def interaction_features(train, test, fea1, fea2, prefix):
    # train : 훈련 데이터
    # test : 테스트 데이터
    # fea1, fea2 : 상호 작용을 수행할 변수 명
    # prefix : 파생 변수명
    
    # 두 변수 간의 곱셈/나눗셈 상호 작용에 대한 파생 변수를 생성한다.
    train['inter_{}*'.format(prefix)] = train[fea1] * train[fea2]
    train['inter_{}/'.format(prefix)] = train[fea1] / train[fea2]
    
    test['inter_{}*'.format(prefix)] = test[fea1] * test[fea2]
    test['inter_{}/'.format(prefix)] = test[fea1] / test[fea2]
    
    return train, test

* 익명회된 데이터를 다루는 경진대회의 경우, 변수 간의 다양한 상호 작용 파생 변수와 
피벗 기반 기초 통계 변수들은 유용한 변수로 작용하는 경우가 자주 있다.
# =============================================================================
# =============================================================================
# FEATURE ENGINEER

# 범주형 변수와 이진 변수 이름을 추출한다.
cat_fea = [x for x in list(train) if 'cat' in x]
bin_fea = [x for x in list(train) if 'bin' in x]

# 결측값(-1)의 개수로 missing 파생 변수를 생성한다.
train['missing'] = (train==1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# 6개 변수에 대해 상호작용 변수를 생성
for e, (x,y) in enumerate(combinations(['ps_car_13','ps_ind_03','ps_reg_03','ps_ind_15','ps_reg_01','ps_ind_01'],2)):
    train, test = interaction_features(train, test, x, y, e)


# 수치형 변수, 상호 작용 파생 변수, ind 변수명을 추출한다.
num_features = [c for c in list(train) if ('cat' not in c and 'calc' not in c)]
num_features.append('missing')
inter_fea = [x for x in list(train) if 'inter' in x]
feature_names = list(train)
ind_features = [c for c in feature_names if 'ind' in c]

# ind 변수 그룹의 조합을 하나의 문자열 변수로 표현한다.
counr = 0
for c in ind_features:
    if count == 0:
        train['new_ind'] = train[c].astype(str)
        count += 1
    else:
        trian['new_ind'] += '_' + train[c].astype(str)
ind_features = [c for c in feature_names if 'ind' in c]
count=0
cor c in ind_features:
    if count == 0:
        test['new_ind'] =test[c].astype(str)
        count += 1
    else:
        test['new_reg'] += '_' + test[c].astype(str)
        
# car 변수 그룹의 조합을 하나의 문자열 변수로 표현한다.
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count==0:
        train['new_car'] = train[c].astype(str)
        count += 1
    else:
        train['new_car'] += '_' + train[c].astype(str)
        
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        test['new_car'] = test[c].astyep(str)
        count += 1
    else:
        test['new_car'] += '_' + test[c].astype(str)
        
# 범주형 데이터와 수치형 데이터를 따로 관리한다.
train_cat = train[cat_fea]
train_num = train[[x for x in list(train) if x in num_features]]
test_cat = test[cat_fea]
test_num = test[[x for x in list(train) if x in num_features]
                
# 범주형 데이터와 수치형 데이터를 따로 관리한다.
max_cat_value =[]
for c in cat_fea:
    le = LabelEncoder()
    x = le.fit_transform(pd.concat([train_cat, test_cat])[c])
    train_cat[c] = le.transform(train_cat[c])
    test_cat[c] = le.transform(test_cat[c])
    max_cat_vlaues.append(np.mean(x))
    
# 범주형 변수의 빈도값으로 새로운 파생 변수 생성
cat_count_features = []
for c in cat_fea + ['new_ind','new_reg','new_car']:
    d = pd.concat([train[c], test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('$s_count'%c)
    
# XGBoost 기반 변수를 읽어온다.
tarin_fea0, test_fea0 = pickle.load(open('./fea0.pk','rb'))

# 수치형 변수의 결측값/이상값을 0으로 대체사고, 범주형 변수와 XGBoost 기반 변수를 통합한다.
train_list = [train_num.replace([np.inf, -np.inf, np.nan],0), train[cat_count_features], train_fea0]
test_list = [test_nem.replace([np.inf, -np.inf, np.nan],0), test[cat_count_features], test_fea0]

# 피벗 기반 기초 통계 파생 변수를 생성.
for t in ['ps_car_13','ps_ind_03','ps_reg_03','ps_ind_15', 'ps_reg_01','ps_ind_01']:
    for g in ['ps_car_13','ps_ind_03','ps_reg_03','ps_ind_15', 'ps_reg_01','ps_ind_01','ps_ind_05_cat']:
        if t != g:
            # group_column 변수를 기반으로 target_column 값을 피벗한 후, 기초 통계 값을 파생 변수로 추가한다.
            s_train, s_test = proj_num_on_cat(train, test, target_column = t, group_column = g)
            train_list.append(s_train)
            test_list.append(s_test)
            
# 데이터 전체를 메모리 효율성을 위해 희소 행렬(sparse.tocsr())로 변환한다.
X = sparse.hstack(tarin_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()
all_data = np.vstack([X.toarray(),X_test.toarray()])
