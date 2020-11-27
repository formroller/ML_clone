# Santander-product-recommendation 

# =============================================================================
# 8등 코드 재현
# =============================================================================
https://github.com/yaxinus/santander-product-recommendation-8th-place

# (참고 사항)
* 데이터 전처리 : 큰 효과를 본 것은 없었지만, 고객 데이터에서 날짜 관련 결측값을 채우려고 시도했으며, 
                 age 변수와 antiguedad 변수를 수정하였고, 
                 lag 5 이상의 데이터를 만들기 위해 2014년도 데이터를 생성하려 하였으나, 소용 없었다.
                 
* 피처 엔지니어링 : 특별할 것이 없고, 평범한 피처 엔지니어링.
                  lag 5, lag데이터에 대한 기초 통계(최솟값, 최댓값, 표준편차)가 주된 피처 엔지니어링이었다.
                  5 이상의 lag데이터로 성능 개선을 보지 못했다.
                  
* 기본 모델 및 모델 통합 : 다양한 변수, 가중치와 모델 설정값을 기준으로 mlogloss 기준으로 학습된 LightGBM과 XGBoost 기본 모델 다수 생성
                         초기 예측값을 기준으로 확률값을 계산하는 알고리즘을 사용해, 초기 예측값 대비 피어슨 상관관계가 가장 낮은 모델 결과물 통합
                         단순히 다수의 모델 결과물 통합하는 것보다, 이 방법이 조금 더 성능 개선을 보였다.

[해당 코드에서 배울 점]
- 데이터 전처리
- 피처 엔지니어링
- 모델 하이퍼파라미터 정의하는 방법
- 머신러닝 파이프라인 구축하는 방법

# =============================================================================
# 데이터 준비
# =============================================================================
 - 8등 팀의 코드는 두 개로 분리된 훈련 데이터와 테스트 데이터를 통합하는 코드로 시작.
 - 훈련 데이터에는 총 48개의 변수(24-고객, 24-금융)가 존재하고,
   테스트 데이터에는 24개의 고객 변수만 존재한다.
 - clean_data()는 존재하지 않는 24개의 금융 변수는 공백으로 대체하는 방식으로 데이터 통합
 
# =============================================================================
# clean.py
# # (코드 2-21) 데이터를 준비하는 코드 : 훈련 데이터와 테스트 데이터 통합
# =============================================================================
 (file : kaggle_santander_product-recommendation/03_winners_code/code/clean.py)
 
# 훈련 데이터와 테스트 데이터를 하나의 데이터로 통합하는 코드
def clean_data(fi,fo,header,suffix):
    
    # fi : 훈련/테스트 데이터를 읽어로는 file iterator
    # fo : 통합되는 데이터가 write되는 경로
    # header : 데이터에 header 줄을 추가할 것인지를 결정하는 boolean
    # suffix : 훈련 데이터에는 48개의 변수 존재, 테스트 데이터에는 24개의 변수만 있어 suffix로 부족한 테스트 데이터 24개분을 공백으로 대체
    
    # csv의 첫 줄(header) 읽어온다.
    head = fi.readline().strip('\n').split(',')
    head = [h.strip('"') for h in head]
    
    # nomprov 변수의 위치를 ip에 저장한다.
    for i,h in enumerate(head):
        if h == 'nomprov':
            ip = i
            
    # header가 True일 경우에는, 저장할 파일의 header를 write한다.
    if header:
        fo.write("%s\n", % ",".join(head))

    # n은 읽어온 변수의 개수를 의미한다 (훈련 데이터:48, 테스트 데이터:24)                
    n = len(head)
    for line in fi:
        # 파일 내용을 한줄 씩 읽어와, 줄바꿈(\n)과 ','으로 분리한다.
        fields = line.strip("\n").split(",")
        
        # 'nomprov'변수에 ','을 포한하는 데이터가 존재한다. ','로 분리된 데이터를 재조합
        if len(fields) > n:
            prov = fields[ip] + fields[ip+1]
            del fields[ip]
            fields[ip] = prov
            
        # 데이터 개수가 n개와 동일한지 확인하고, 파일에 write한다. 테스트 데이터의 경우, suffix는 24개의 공백이다.
            assert len(fields) == n 
            fields = [field.strip() for field in fields]
            fo.write("%s%s\n" % (",".join(fields), suffix))
            
# 하나의 데이터를 통합하는 코드를 실행한다. 먼저 훈련 데이터를 write하고, 그 다음 테스트 데이터를 write한다.
with open("./8th.clean.all.csv","w") as f:
    clean_data(open('./train_ver2.csv', f, True,""))
    comma25 = "".join([","for i in range(24)])
    clearn_data(open('./test_ver2.csv'), f ,False, comma24)


# =============================================================================
# main.py
## main.py에서 필요한 라이브러리를 불러오는 코드
# =============================================================================
(main.py)

해당 파일(main.py)에서 1)데이터 전처리 - 2) 피처 엔지니어링 - 3) 머신러닝 모델 학습 - 4) 테스트 데이터 예측 및 캐글 업로드 
일련의 머신러닝 파이프라인 과정을 모두 수행한다.

main.py에서는 먼저 데이터 전처리와 피처 엔지니어링에 필요한 라이브러리를 불러온다.
관련 라이브러리와 머신러닝 학습 함수가 포함된 engines.py, 자주 사용되는 함수가 포함된 utils.py로 함께 임포트한다.

import math
import io

# 파일 압축 용도
import gzip
import pickle
import zlib

# 데이터, 배열을 다루기 위한 기본 라이브러리
import pandas as pd
import numpy as np

# 범주형 데이터를 수치형으로 변환하기 위한 전처리 도구
from sklearn.preprocessing import LabelEncoder

import engines
from utils import *

np.random.seed(2016)
transformers = {}

main.py 함수에서 데이터 전처리와 피처 엔지니어링은 make_data()가 담당하고, 
머신러닝 모델 학습 및 캐글 제출용 파일 생성은 train_predict()에서 담당한다.

## 1. 데이터 전처리 - 2. 피처 엔지니어링
make_data()에서는 데이터 전처리와 피처 엔지니어링을 굳이 별도의 파이프라인으로 분리하지 않고, 동시에 수행.
이번 경진대회에서 가장 핵심적인 역할을 한, 데이터 전처리와 피처 엔지니어링을 수행하는 make_data()는
먼저 load_data()를 통해 데이터를 읽어온 후, 전처리 / 피처 엔지니어링을 수행한다.

# (코드 2-23) 제품 변수에 대한 결측값을 대체하고, 전처리 / 피처 엔지니어링을 수행한다.
# main.py line 152
# '데이터 준비'에서 통합한 데이터를 읽어온다.
fname = '../input/8th.clean.all.csv'
train_df = pd.read_csv(fname, dtype = dtypes)

# products는 util.py에서 정의한 24개의 금융 제품 이름이다.
# 결측값을 0.0으로 대체하고, 정수형으로 변환한다.
for prod in products:
    train_df[prod] = train_df[prod].fillna(0.0).astype(np.int8)
    
# 48개의 변수마다 전처리/피처 엔지니어링을 적용한다.
train_df, features = apply_transforms(train_df)

통합 데이터에서 24개의 금융 변수에 대해 결측값을 대체하고, 정수형으로 변환한 후,
apply_transforms()를 사용해 24개의 고객 변수에 대해 데이터 전처리 및 피처 엔지니어링을 1차적으로 수행한다.

# apply_transform()함수 설명 전, 이 함수에 포함된 4개의 도구 함수들에 대한 설명


도구 1) label_encode(df, features, name)함수는 데이터 프레임(df)에서 범주형 변수 name을 LabelEncoder()를 사용해 수치형으로 변환,
   사전에 정의한 dict()인 transformers에 label encoding을 수행한 변수명을 기록해,
   df에 동일한 변수를 label encoding 할 때에는 기존에 LabelEncoder()를 재활용한다.
   (실제 코드에서는 transformers는 한 번도 재사용되지 않는.)
   
# (코드 2-24) 범주형 변수를 수치형으로 변환하는 label_encode 함수
# main.py line 34
def label_encode(df,features,name):
    # df의 변수 name의 값을 모두 string으로 변환한다.
    df[name] = df[name].astype('str')
    # 이미, label_encode했던 변수일 경우, transformer[name]에 있는 LabelEncoder()를 재활용한다.
    
    if name in transformers:
        df[name] = transformers[name].transform(df[name])
    # 처음 보는 변수일 경우, transformer에 LabelEncoder()를 저장하고, .fit_transform() 함수로 label encoding을 수행한다.
    else:
        transformers[name] = LabelEncoder()
        df[name] = transformers[name].fit_transform(df[name])
    # label encoding한 변수는 features 리스트에 추가한다.
        
도구 2) encode_top(s, count=100, dtype = np.int8)함수는 pd.Series에서 빈도가 가장 높은 100개의 고유값을 순위로 대체하고,
    그 외, 빈도가 낮은 값을 모두 0으로 변환한 로운 pd.Series를 반환한다.
    데이터 전체가 아닌, 고빈도 데이터에 대한 정보를 추출하는 함수이다.
    
# (코드 2-25) 빈도 상위 100개의 데이터에 대한 순위 변수를 추출하는 함수
# main.py line 47
def encode_top(s, count = 100, dtype=np.int8):
    # 모든 고유값에 대한 빈도를 계산한다.
    uniqs, freq = np.unique(s, return_counts=True)
    # 빈도 top100을 추출한다.
    top = sorted(zip(uniqs, freqs), key = lambda vk:vk[1], reverse=True)[:count]
    # {기존 데이터 : 순위}를 나타내는 dict()를 생성한다.
    top_map = {uc[0] : l+1 for uf, l in zip (top, range(len(top)))}
    # 고빈도 100개의 데이터는 순위로 대체하고, 그 외는 0으로 대체한다.
    return s.map(lambda x : top_map.get(x,0)).astype(dtype)

도구 3) date_to_float(str_date) 함수는 입력으로 들어오는 날짜를 숫자로 변환하는 함수.
    입력값이 문자열 형태의 날짜 데이터일 경우[년도*12+월]이라는 계산으로 날짜 데이터를 소수로 환산해 반환한다.
    date_to_int(str_date)는 월 단위 수치형으로 변환된 데이터를 1-18 사이의 값으로 변환한다.
# (코드2-26) 날짜 데이터를 숫자로 변환하는 두 가지 함수(utils.py)
# utils.py line 23
# 날짜 데이터를 월 단위 숫자로 변환하는 함수
def date_to_float(str_date):
    if str_date.__class__ is float and math.isnan(str_date) or str_date = "":
        return np.nan
    Y,M,D = [int(a) for a in str_date.strip().split("-")]
    float_date = float(Y) * 12 + float(M)
    return float_date

# 날짜 데이터를 월 단우 숫자로 변환하되 1-18사이로 제한하는 함수
def date_to_int(str_date):
    Y,M,D = [int(a) for a in str_date.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 +int(M)
    assert 1 <= int_date <= 12 + 6 # (assert, 가정 설명문) 뒤의 조건이 Fasle일 경우 assertError발생한다
    return int_date

도구 4) custom_one_hot(df, features, name, names) 함수는 범주형 변수를 입력으로 받아, 변수 안에 존재하는 고유값을 새로운 이진 변수로 생성하는 one-hot-encoindg수행.
    범주형 데이터를 하나의 열에서 label encoding하는 것보다 표현력이 높아지지만, 고유값의 숫자만큼 데이터의 열이 늘어나기 때문에, 고유값이 적은 데이터에서 선호되는 피처 엔지니어링 기법.
    sklearn.preprocessing.OneHotEncoding, pandas.get_dummies gkatnsms one-hot-encoding을 지원하지만, 여기서는 직접 구축했다.
# (코드 2-27) 자체 구현한 one-hot-encoding
# main.py line26
def custom_one_hot(df, features, name, names, dtype=np.int8, check = False):
    for n, val in names.item():
        # 신규 변수명을 "변수명_숫자"로 지정한다.
        new_name="%s_%S" & (name, n)
        # 기존 변수에서 해당 고유값을 가지면 1, 그 외는 0인 이진 변수를 생성한다.
        df[new_name] = df[name].map(lambda x : 1 if z == val else 0).astype(dtype)
        features.append(new_name)
    
## apply_transform()에서는 앞에서 설명한 4개의 도구 함수를 사용해 총 48개의 변수에 대한 데이터 전처리와 피처 엔지니어링을 수행한다.
## 주된 데이터 전처리와 피처 엔지니어링 내용은 다음과 같다.
- 결측값 대체(0.0 or 1.0)
*범주형 데이터 label encoding
-  범주형으로 표현되는 데이터를 sklearn.preprocessing의 LabelEncoder 도구를 사용해 수치형으로 변환.
*고빈도 top 100ro를 빈도 순위로 변환 
- 특정 변수에서 빈도가 높은 값을 순위로 변환해, 고빈도 데이터에 대한 선형 관계를 추출.
*수치형 변수 log transformation 
- log transformation은 데이터 내의 대소관계를 유지하며 포함된 값들의 차이를 줄여주는 역할을 한다.
*날짜 데이터에서 년/월을 추출 
- '2015-06-28'과 같은 문자열 데이터에서 연도와 월을 추출한다.
*날짜 데이터 간의 차이값으로 파생 변수 생성
- 2개의 날짜 데이터의 차이값을 통해 상대적인 거리 변수 생성.
*one-hot-encoding 변수 생성 
 - 범주형 데이터의 표현력을 높이기 위해, 모든 고유값을 새로운 이진 변수로 생성한다.
 
# (코드 2-28) 데이터 전처리와 피처 엔지니어링 일부를 수행하는 apply_transform 함수
# main.py line 57
def apply_transforms(train_df):
    
    # 학습에 사용할 변수를 저장할 features 리스트 생성
    features = []
    
    # 두 변수를 label_encode() 한다.
    label_encode(train_df, features, 'canal_entrada')
    label_encode(train_df, features, 'pais_residencia')
    
    # age의 결측값을 0.0으로 대체하고, 모든 값을 정수로 변환한다.
    train_df['age'] = train_df['age'].fillna(0.0).astype(np.int16)
    features.append('age')
    
    # renta의 결측값을 1.0으로 대체하고, 모든 값을 정수로 변환한다.
    train_df['renta'] = train_df['age'].fillna(1.0, inplace = True)
    train_df['renta'] = train_df['renta'].map(math.log)
    reatures.append('renta')
    
    # 고빈도 100개의 순위를 추출한다.
    train_df['renta_top'] = encode_top(train_df['renta'])
    reatures.append('renta_top')
    
    # 결측값 혹은 음수를 0으로 대체하고, 나머지 값은 +1.0 한 뒤, 정수로 변환
    trian_df['antiguedad'] = train_df['antiguedad'].map(lambda x ; 0.0 if x < 0 or math.isnan(x) else x+1.0).astype(np.int16)
    features.append('antiguedad')
    
    # 결측값을 0.0으로 대체하고, 정수로 변환한다.
    train_df['tipodom'] = train_df['tipodom'].fillna(0.0).astype(np.int8)
    features.append('tipodom')
    train_df['cod_prov'] = train_df['cod_prov'].fillna(0.0).astype(np.int8)
    features.append('cod_prov')
    
    # fecha_dato에서 월/년도를 추출해 정수값으로 변환한다.
    train_df['fecha_dato_month'] = train_df['fecha_dato'].map(lambda x :int(x.split('-')[1])).astype(np.int8)
    features.append('fecha_dato_month')
    train_df['fecha_dato_year'] = train_df['fecha_dato'].map(lambda x :int(x.split('-')[0])).astype(np.int8)
    features.append('fecha_dato_year')
    
    # 결측값을 0.0으로 대체하고, fecha_alta에서 월/년도 추출하여 정수값으로 변환한다.
    # x.__class__는 결측값일 경우 float를 반환하기 때문에, 결측값 탐지용으로 사용하고 있다.
    train_df['fecha_alta_month'] = train_df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float (x.split('-')[1])).astype(np.int8)
    features.append('fecha_alta_month')
    train_df['fecha_alta_year'] = train_df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float (x.split('-')[0])).astype(np.int8)
    features.append('fecha_alta_year')
    
    # 날짜 데이터를 월 기준 수치형 변수로 변환
    train_df['fecha_dato_float'] = train_df['fecha_dato'].map(date_to_float)
    train_df['fecha_alta_float'] = train_df['fecha_alta'].map(date_to_float)
    
    # fecha_dato와 fecha_alta의 월 기준 수치형 변수의 차이값을 파생 변수로 생성한다.
    train_df['dato_minus_alta'] = train_df['fecha_dato_float'] - train_df['fecha_alta_float']
    features.append('dato_minus_alta')
    
    # 날짜 데이터를 월 기준 수치형 변수로 변환한다.(1-18 사이 값으로 제한)
    train_df['int_date'] = train_df['fecha_dato'].map(date_to_int).astype(np.int8)
    
    # 자체 개발한 one-hot-encoding을 수행
    custom_one_hot(train_df, features, 'indresi', {"n":"N"})
    custom_one_hot(train_df, features, 'indext', {"s":"S"})
    custom_one_hot(train_df, features, 'conyuemp', {"n":"N"})
    custom_one_hot(train_df, features, 'sexo', {"h":"H", "v":"V"})
    custom_one_hot(train_df, features, 'ind_empleado', {"a":"A","b":"B","f":"F","n":"N"})
    custom_one_hot(train_df, features, 'ind_nuevo', {"new":1})
    custom_one_hot(train_df, features, 'segmento', {"top":"01-TOP","particulares":"02-PARTICULARES","universitario":"03-UNIVERSITARIO"})
    custom_one_hot(train_df, features, 'indfall', {"s":"S"})
    custom_one_hot(train_df, features, 'indrel', {"1":1, "99":99})
    custom_one_hot(train_df, features, 'tiprel_1mes', {"a":"A","i":"I","p":"P","r":"R"})
    
    # 결측값을 0.0으로 대체하고, 그 외는 +1.0 덧셈 후 정수로 변환
    train_df['ind_actividad_cliente'] = train_df['ind_actividad_cliente'].map(lambda x : 0.0 if math.isnan(x) else x+1.0).astype(np.int8)
    features.append('ind_actividad_cliente')

    # 결측값을 0.0으로 대체하고, "P"를 5로 대체 한 뒤, 정수로 변환.
    train_df['indrel_1mes'] = train_df['indrel_1mes'].map(lambda x : 5.0 if x == "P" else x).astype(float).fillna(0.0).astype(np.int8)
    features.append('indrel_1mes')
    
# 데이터 전처리/피처 엔지니어링이 1차적으로 완료된 데이터 프레임 train_df와 학습에 사용할 변수 리스트 features를 tuple 형태로 반환한다.
return train_df, tuple(features)

# 데이터 전처리 및 피처 엔지니어링을 1차적으로 완료한 데이터를 train_df에 저장하고, 학습에 사용할 변수 목록을 features에 저장한다.
train_df, features = apply_transforms(train_df)


## -----
다음은 금융 변수 lag 데이터를 생성한다. 시계열 문제에서 많이 사용되는 파생 변수의 하나로써, 해당 변수의 n단위 이저느이 값을 lag-n 데이터라 한다.
예를 들어, 1달 전 고객 등급을 현재 시점으로 끌어와 '고객등급_lag_1'이라는 새로운 파생 변수로 활용할 수 있다.
(시계열 경진대회에서는 lag-n 데이터가 유의미한 성능 개선을 보이는 경우가 종종 있다.**)

lag 변수를 생성하기 위한 2개 도구 함수 설명
도구 1) make_prev_df(train_df, step) 함수는 24개의 금융 변수에 대한 lag 데이터르 직접 생성하는 함수이다.