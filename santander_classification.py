# =============================================================================
# chapter2. 산탄데르 제품 추천 경진대회
# =============================================================================
- Santander Product Recommandation Competition

# 2-1. 경진대회 소개
주최자     :  산탄데르 은행
문제 유형  : Multi-class Classification(다중 클래스 분류)
평가 척도  : Mean Average Precision @7
대회 목적  : 고객이 신규로 구매할 제품이 무엇인지 예측

# 2-2. 경진 주최자 동기
아직 고객이 사용하고 잇지 않은 다른 금융 제품을 소개해 고객의 만족도를 높임과 동시에 은행 매출을 올리고자함.
이를 위해 캐글 플랫폼의 힘을 빌려 단기간에 유의미한 알고리즘을 개발이 목적.

# 2-3. 평가 척도
MAP@7 (Mean Average Precision @7), 예측 정확도의 평균

1) Mean Average : 정확도의 합을 정답의 갯수만큼 나눈 숫자.
 ## 기존 MA
 #Precision(예측 결과)
  1 0 0 1 1 1 0  (0=오답, 1=정답)
 #Precision(예측 정확도)
  1/1 0 0 2/4 3/5 4/6 0
 #Average Precision(예측 정확도의 평균)
  (1/1 + 2/4 + 3/5 + 4/6)/4 = 0.69 *정답의 갯수인 4로 나눔.  
 ##MAP @7
  : 모든 예측 결과물의 Average Precision의 평균값을 의미.
  - @7 -> 최대 7개의 금융 제품을 예측할 수 있다는 것을 의미
  - 예측의 순서에 매우 예민한 평가 척도.(=정답을 앞쪽에 예측하는 것이 더 좋은 점수를 받을 수 있다.)
  
  (경진대회에서 MAP@7 평가 척도를 구하기 위해서는 다음과 같은 코드 사용.)
mapk()의 입력 값으로 들어가는 actual, predicted는 (고객 수 * 7)의 dimension을 갖는 list of list이다.
7개의 금융 제품명이 숫자 목록 형태로 저장되고, 이러한 목록이 고객의 수만큼 있는 list of list이다.

import numpy as np 

def apk(actual, predicted, k=7, default=0.0):
    #map@7이므로, 최대 7개만 사용한다.
    if len(predicted) > 7:
        predicted = predicted[:k]
        
        score = 0.0
        num_hits=0.0
        
        for i, p in enumerate(predicted):
            # 점수를 부여하는 조건은 다음과 같다:
            # 예측 값이 정답에 있고('p in actual)
            # 예측 값이 중복이 아니면 ('p not in predicted[:i]')
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
                
       # 정답 값이 공백일 경우, 무조건 0.0점을 반환한다.
        if not actual:
            return default
       # 정답의 개수(len(acutal))로 average Precisin을 구한다.
        return score/min(len(actual),k)
            

# 2.4 주요 접근
산탄데르 제품 추천 경진대회에서는 Tabular 형태의 시계열(Time-Series)데이터가 제공되었다.

# A.데이터 전처리 작업이 필수
- NA or nan
- Outlier
- 사람의 실수로 잘못 입력된 값
- 개인 정보를 위해 익명화된 변수 등..

# B. 피처 엔지니어링 필수
- 시계열 데이터를 다루는 경진대회에서는 유의미한 파생 변수를 생성하기 위한 몇 가지 기술들이 있다.
- 날짜/시간, 주중/주말, 아침/낮/밤,..
-> 시계열 데이터에서는 과거 데이터를 활용하는 lag데이터를 파생 변수로 활용할 수 있다.

* Boosting Tree 모델이란,
데이터를 하나의 트리 모델에 학습 시킨 후, 
해당 트리 모델의 성능이 낮은 부분을 보완하는 다른 트리 모델을 학습시키는 방식으로,
수 많은 트리 모델을 순차적으로 학습 시키며 성능을 개선하는 모델이다.
하나의 Boosting Tree 모델에 수십, 수백, 수천 개의 트리 모델이 사용된다.

import pandas as pd
import numpy as np
import os
os.getcwd()
os.chdir('C:\\Users\\yongjun\\.spyder-py3\\kaggle\\book')

trn = pd.read_csv('./santander/train_ver2.csv')

trn = pd.read_csv('train_ver2.csv', skiprows=range(15000,13647306))


trn.shape
trn.head()
# 모든 변수에 대해 미리보기 실행
for col in trn.columns:
    print('{}\n'.format(trn[col].head()))
    
# fecha_dato : 날짜를 기록하는 fecha_dato 변수는 날짜 전용 data type: datetime이 아닌 object다.
# age : 고객으 나이를 기록하는 age 변수는 정수값을 가지는 것 같지만 data type이 아닌 object다.(전처리 과정에서 정수값으로의 변환이 필요.)
# renta : 가구 총수입을 나타내는 renta 변수는 소수값을 가지며, 5번째 열에서는 NaN값을 갖는다.

trn.info()
# 변수 
1-24 : 고객 관련 변수
25-48 : 금융 제품 변수
# 수치형 / 범주형 변수
첫 24개의 고객 변수 중, ['int64','float64']의 데이터 타입을 갖는 수치형 변수를 num_cols로 추출하고, 
.describe()를 통해 간단한 요약 통계를 확인한다.

num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64','float64']]
trn[num_cols].describe()

['ncodpers', 'ind_nuevo', 'indrel', 'tipodom', 'cod_prov','ind_actividad_cliente', 'renta']
=> 24개의 변수중 7개 변수 수치형 변수이다.

* ncodpers : 최솟값 15889 - 최댓값 1553689을 갖는 고유 식변 번호이다.
* ind_nuevo : 최소 75%값이 1 이며, 나머지가 1을 갖는 신규 고객 지표.
* indrel : 최소 75%의 값이 1 이며, 나머지가 99를 갖는  고객 등급 변수.
* tipodom : 모든 값이 1인 주소 유형 변수. 이러한 변수는 학습에 도움이 되지 않는 변수이다.
            모든 값이 상수일 경우에는 변수로서 식별력을 가질 수 없기 때문이다.
* cod_prov : 최소 1 ~ 최대 52의 값을 가지며, 수치형이지만 범주형 변수로서 의미를 가지는 지방 변수 코드이다.
* ind_actividad_cliente : 최소 50%의 값이 0이며, 나머지가 1을 갖는 활발성 지표이다.
* renta : 최소 1202.73 ~ 최대 28894400의 값을 갖는 전형적인 수치형 변수, 가구 총수입을 나타낸다.

이번에는 첫 24개의 고객 변수 중, ['object']의 데이터 타입을 갖는 범주형 변수를 cat_cols로 추출하고, 
.describe()를 통해 간단한 요약 통계를 확인한다.

cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]
trn[cat_cols].describe()

24개의 고객 변수 중, 나머지 17개의 변수가 범주형 변수이다.

범주형 변수에 대한 결과값의 각 행은 다음과 같은 의미가 있다.

* count : 해당 변수의 유효한 데이터 개수를 의미한다.
          13,647,309보다 작을 때, 그만큼의 결측값이 존재한다는 뜻이다.
          특히 'ult_fec_cli_1t'는 count가 24,793 밖에 확인되지 않으며, 결측값이 대부분임을 확인할 수 있다.
* unique : 해당 범주형 변수의 교유값 갯수를 의미한다. 성별 변수 'sexo'에는 고유값이 2개이다.
* top : 가장 빈도가 노은 데이터가 표시된다. 나이 변수 'age'에서 최빈 데이터는 23세이다.
* freq : top에서 표시된 최빈 데이터의 빈도수를 의미한다. 총 데이터 수를 의미하는 count대비 최빈값이 어느 정도인지에 따라 범주형 데이터의 분포를 가늠할 수 있다.
         예를 들어 고용 지표 변수인 'ind_empleado'는 5개의 고유값 중, 가장 빈도가 높은 'N' 데이터가 전체의 99.9%가량 차지하며, 데이터가 편중됨을 알 수 있다.
         여기서 주의해야 할 것이 하나 있다.
         나이를 의미하는 'age' 변수가 수치형이 아니라 범주형으로 분류되어 있다는 것이다. 그 외, 은행 누적 거래 기간을 나타내는 
         'antiguedad'도 수치형으로 분류되어야 한다. 원본 데이터에는 이와 같이 수치형 데이터가 범주형/objec로 분류되어 있기 때문에, 전처리 과정에서 수치형으로 변환해야 한다.
         
# 범주형 변수의 고유값을 직접 눈으로 확인해보자. 다음의 코드를 통해 17개의 cat_cols에 포함된 고유값을 직접 출력한다.
for col in cat_cols:
    uniq = np.unique(trn[col].astype(str))
    print('-' * 50)
    print('# col {}, n_uniq{}, uniq{}'.format(col, len(uniq), uniq))
    
# 1) fecha_dato : 날짜를 기록하는 frcha_dato는 '년-월-일' 형태이다.
                 2015-01-28 ~ 2016-05-28까지 매달 28일에 기록된 월별 데이터 임을 확인할 수 있다.
# 2) ind_empleado : 5가지 값과 결측값인 na를 포함하고 있다.
# 3) pais_residencia : 고객 거주 국가를 나타내는 해당 변수는 알파벳 두 글자 형태를 취하고 있다.
# 4) sexo : 남/여/결측값
# 5) age : 105세 익상 이상값인지 확인 필요
# 6) antiguedad : -999999값 출력 (이상치), 공백 포함
# 7) ult_fec_cli_1t : nan ~ 16)segmento nan 포함

# 데이터 분석 노트
변수명         내용           데이터 타입          특징                          변수 아이디어
fecha_dato    월별 날짜       object                                            년도, 월 데이터를 별도로 추출

ncodpers      고객 고유 번호  int 64             숫자로 되어있으나, 
                                                엄밀히 식별 번호이다

ind_empleado  고용 지표       object            5개 고유값 중 'N'이99.9%
                                                빈도 편중이 높아 변수 중요도 낮을 것
                                                
pais_residencia 고객 거주 국가 object           알파벳 두글자로 생성된 국가 변수이나,
                                               암호화 되어 있어 국가 역추적은 어렵다.

sexo           성별            object           성별 변수이다.

age            나이            object                                           나이 데이터가 정수형이 아니다.(정제 필요)
                             -> int64   

fecha_alta     고객이 은행과   object           95-16년까지 폭넓은 값을 갖는다.
              첫계약을 체결한 날짜              장기 고객도 존재하는 것으로 보임   
             
ind_nuevo     신규 고객 지표   float64          대부분이 0, 소수가 1인 변수.     정수로 변환
             (6개월 이내 신규  ->int64
             고객일 경우 값=1)  
             
antiguedad    은행 거래        object 
             누적 기간(월)    ->int64
             
indrel       고객 등급         float64          대부분이 1, 소수가 99인 변수     정수로 변환
           (1:1등급 고객,      ->int64 
            99:해당 달에
            1등급 해제되는
            1등급 고객)

ult_fec_li_lt 1등급 고객으로  object             2015-7 ~ 2016-5월까지 데이터
              마지막 날짜

indrel_1mes   월초 기준 고객 등급 object        '1'과 '1.0'이 다른값으로 존재    'P'를 정수로 변환하고 '1','1.0' ->1로 변환  
                                 ->int64 
                                 
...
tipodom       주소유형        float64           모든 값이 1. 변수로서 무의미하다. 변수를 제거하자.

cod_prov      지방 코드       float64                                           정수로 변환   
                             -> int64

ind_actividad_cliente        float64                                            정수로 변환 
                             -> int64 

renta          가구 총 수입   float64                                           정수로 변환
                             -> int64
                             
                        
# 시각화로 데이터 살펴보기 (ncodpers, renta제외-> 고유값이 너무 많아 시간 소요)
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns

skip_cols = ['ncodpers','renta']
for col in trn.columns:
    # 출력에 너무 시간이 많이 걸리는 두 변수는 skip한다.
    if col in skip_cols:
        continue
    
    # 보기 편하게 영역 구분과 변수명을  출력한다.
    print('-'*50)
    print('col ; ',col)
    
    # 그래프 크기를(figsize) 설정한다.
    f, ax = plt.subplots(figsize=(20,15))
    # seaborn을 사용한 막대 그래프를 생성
    sns.countplot(x=col,data=trn,alpha=0.5)
    #show() 함수를 통해 시각화
    plt.show()
    
# 시계열 데이터 시각화(누적 막대그래프)
누적 막대그래프 => 서로 다른 제품 간의 차이를 함께 시각화하기 위함이다.

# 월별 금융 제품 보유 여부를 누적 막대 그래프로 시각화하는 코드

# 날짜 데이터를 기준으로 분석하기 위해, 날짜 데이터 별도로 추출한다
months = trn['fecha_dato'].unique().tolist()
# 제품 변수 24개를 추출한다.
label_cols = trn.columns[24:].tolist()

label_over_time=[]
for i in range(len(label_cols)):
    # 매월, 각 제품의 총합을 groupby(..).add('sum')으로 계산하여, label_sum에 저장한다.
    label_sum = trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
    label_over_time.append(label_sum.tolist())
    
label_sum_over_time=[]
for i in range(len(label_cols)):
    # 누적 막대 그래프를 시각화하기 위해, n번째 제품의 총합을 1 ~ n번째 제품의 충합으로 만든다.
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
   
# 시각화를 위해 색 지정
color_list = ['#F5B7B1', '#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']

# 그림 크기 사전에 지정
f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    # 24개 제품에 대해 histogram 그린다.
    # x축에는 월 데이터, y축에는 누적 총합, 색은 8개를 번갈아 사용하며 그림의 alpha값은 0.7로 지정
    sns.barplot(x=months, y=label_sum_over_time[i], color=color_list[1%8], alpha=0.7)

# 우측 상단에 Legend 추가
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor='none')for i in range(len(label_cols))], label_cols, loc=1, ncol=2,prop={'size':16})


##(코드2-11) 월별 금융 제품 보유 데이터를 누적 막대 그래프로 시각화하기 : 절댓값이 아닌 월별 상댓값으로 시각화하여 시각적으로 보기 쉽게 표현한다.

# label_sum_over_time의 값을 퍼센트 단위로 변환한다. 월마다 최댓값으로 나누고 100을 곱해준다.
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0)))*100
# 앞선 코드와 동일한, 시각화 실행 코드이다.
f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
    
plt.legend([plt.Rectangle((0,0), 1, 1, fc=color_list[i%8], edgecolor='none') for i in range(len(label_cols))], label_cols, loc=1, ncol=2,prop={'size':16})


# (코드2-12) 24개의 금융 제품에 대한 '신규 구매' 데이터 생성
# 제품 변수를 prods에 list형태로 저장한다.
prods = trn.columns[24:].tolist()
# 날짜를 숫자로 변환하는 함수 (2015-01-28은 1, 2016-06-28은 18로 변환된다.)
def date_to_int(self):
    Y,M,D = [int(a) for a in self.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date
# 날짜를 숫자로 변환해 int_date에 저장한다.
trn['int_date'] = trn['fecha_dato'].map(date_to_int).astype(np.int8)
#데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가(메모리 부족)
trn_lag = trn.copy()
trn_lag['int_date'] += 1
trn_lag.columns = [col + '_prev' if col not in['ncodpers','int_date'] else col for col in trn.columns]
#원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다. lag 데이터의 int_date는 1 밀려 있기 때문에, 지난 달의 제품 정보가 삽입
df_trn = trn.merge(trn_lag, on=['ncodpers','int_date'], how = 'left')

# 메모리 효율을 위해 불필요한 변수를 메모리에서 제거한다.
del trn, trn_lag
# 저번 달의 제품 정보가 존재하지 않을 경우를 대비해 0으로 대체한다.
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0,inplace=True)
# 원본 데이터에서의 제품 보유 여부 -lag 데이터에서 제품 보유 여부를 비교하여 신규 구매 변수 padd를 구한다.
for prod in prods:
    padd = prod + '_add'
    prev = prod + '_prev'
    df_trn[padd] = ((df_trn[prod]==1) & (df_trn[prev] == 0)).astype(np.int8)
    
## 신규 구매 변수만을 추출하여 labels에 저장한다.
add_cols = [prod + '_add' for prod in prods]
labels = df_trn[add_cols].copy()
labels.columns = prods
labels.to_csv('./labels.csv', index=False)

##월별 신규 규매 데이터를 누적 막대 그래프로 시각화하기.
labels = pd.read_csv('./labels.csv').astype(int)
fecha_dato = pd.read_csv('./train_ver2.csv',usecols=['fecha_dato'], skiprows=range(15000,13647306))

labels['date'] = fecha_dato.fecha_dato
months = np.unique(fecha_dato.fecha_dato).tolist()
labels_cols = labels.columns.tolist()[:24]

label_over_time=[]
for i in range(len(labels_cols)):
    label_over_time.append(labels.groupby(['date'])[label_cols[i]].agg('sum').tolist())

label_sum_over_time = []
for i in range(len(label_cols)):
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
    
color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE', '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']

f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha = 0.7)
    
plt.legend([plt.Rectangle((0,0), 1, 1, fc=color_list[i%8], edgecolor = 'none'), for i in range(len(label_cols))], label_cols, loc=1, ncol=2, prop={'size':16})
# 첫 달인 2015-05-28의 신규 구매 숫자가 높다? -> 첫 달에는 모든 보유 제품이 신규 구매로 인식되기 때문!


##(코드2-14) 월별 신규 구매 데이터를 누적막대 그래프로 시각화하기 : 절댓값이 아닌 월별 상대값으로 시각화하여 시각적으로 보기 쉽게 표현한다.
#(코드 2-11과 동일한 코드)
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100
f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha =0.7)
    
plt.legned([plt.Rectangle((0,0),1,1, fc=color_list[i%8], edgecolor=('none') for i in range(len(label_cols))], label_cols, loc=1, ncol=2, prop={'size':16}))

#흥미로운 패턴
* 당좌 예금(ind_cco_fin_utl1, 상위 첫번째)은 8월(여름)에 가장 높은 값을 가지며, 겨울에는 축소되는 계절 추이를 갖는다.
* 단기 예금(ind_deco_fin_ult1, 상위 다섯번째)은 2015-06-28에 특히 높은 값을 가지며 다른시기에는 값이 매우 낮다.
* 급여, 연금(ind_nomina_ult1, int_nom_pens_ut1)은 당좌 예금과 반대로 8월 여름에 가장 잔은 값을 가지며 2016-02-28 겨울에 가장 높은 값을 갖는 추세
* 신규 구매 빈도가 가장 높은 상위 5개 금융 제품은 당좌 예금, 신용 카드, 급여, 연금 , 직불 카드이다.(cco_fin, tjct_fin, nomina, nom_pens, recibo)

=> 데이터가 계절성을 띈가는 것은, 훈련 데이터를 몇 월로 지정하는가에 따라, 모델의 결과물이 많이 달라질 수 있다는 것을 의미한다.
   계절의 변동성을 모델링하는 하나의 일반적인 모델을 구축할 것인지, 계절에 따라 다수의 모델을 구축하여 혼합해 사용할지 결정해야 한다.
   (실무에서는 다수의 모델을 계절별로 구축해 얻는 성능의 개선폭과, 다수의 모델을 실시간으로 운영하는 비용 및 리스크를 비교하는 것이 필요.)
   
   
# 탐색적 데이터 분석 요약
   1) 기초 통계를 통해 Raw Data 분석하는 방법
   2) 시각화 통한 Raw Data 분석
# 훈련 / 테스트 데이터 설명
이번 경진대회에서는 총 1년 6개월치(2015-01-28 ~ 2016-06-28) 월별 고객 데이터가 제공된다.

첫 1년 5개월치(2015-01-28 ~ 2016-05-28)데이터는 훈련 데이터이며, 훈련 데이터에는 익명화된 24개의 고객 변수와 25개의 금융 제품 보유 현황에 대한 정보 포함.

[데이터 일차적으로 살펴본 결과 요약.]
* 'age','antiguedad','indres_1mes'등의 수치 변수가 object로 표현돼 인식 불가.(정제 작업 필요)
* 대부분 고객 변수에 결측값 존재. 수치/범주형 변수의 결측값은 기존에 없는 값(흔히 0,-1등을 사용)으로 흔히 대체한다. (날짜 변수에 대한 고찰 필요)
* 이진 변수 다수 존재, 이를 int64의 0,1 값으로 변환.
* 구객 등급, 관계 유형 등 변수값의 설명 부족.
* 예측하고자 하는 값 : '신규 구매'
  -> 제공된 데이터에서 '신규 구매' 여부를 별도로 추출해야 하며, 평가 기준도 '신규 구매' 기준으로 진행해야 한다.(label.csv)
* 신규 구매 데이터가 계절성을 띄고 있다.
  -> 단일 모델로 모든 데이터를 학습시킬지, 특정 월만 추출해서 학습을 진행할지 선택 필요
     다수의 모델을 서로 다른 계절 기반으로 학습하는 것도 하나의 방법이다.
    
마지막으로 한달치(2016-06-28) 데이터는 테스트 데이터로 사용되며, 24개의 고객 변수는 동일하게 제공되나, 금융 제품 보유 현황에 관련한 값이 존재하지 않는다.

# =============================================================================
# # 2.7 Baseline 모델
# =============================================================================
Baseline모델은 일반적인 머신러닝 파이프하인의 모든 과정을 포함하는 가장 기초적인 모델이다.

(머신러닝 파이프라인의 일반적인 순서)
1) 데이터 전처리 -> 2) 피처 엔지니어링 -> 3) 머신러닝 모델 학습 -> 4) 테스트 데이터 예측 및 캐글 업로드
            
# =============================================================================
# 1) 데이터 전처리
# =============================================================================
Baseline 모델 구축을 위한 데이터 전처리 과정에서는 다음 작업을 수행한다.

* 제품 변수의 결측값을 0으로 대체한다.
  - 제품 보유 여부에 대한 정보가 없으면, 해당 제품을 보유하고 있지 않다고 가정한다.

* 훈련 데이터와 테스트 데이터를 통합한다.
  - 훈련 데이터와 테스트 데이터는 날짜 변수(fecha_dato)로 쉽게 구분이 가능하다.
  - 동일한 24개의 고객 변수를 공유하며, 테스트 데이터에 없는 24개의 제품 변수는 0으로 채운다.
  
* 범주형, 수치형 데이터를 전처리한다. 
  - 범주형 데이터는 Label Encoding을 수행한다.
  - 데이터 타입이 object로 표현되는 수치형 데이터에서는 .unique()를 통해 특이값들을 대체하거나 제거하고, 정수형 데이터로 변환한다.

* 추후, 모델 학습에 사용할 변수 이름을 feature 리스트에 미리 담는다.

# (코드2-15)Baseline 모델의 전처리 코드

import os
os.getcwd()
os.chdir('..spyder-py3/kaggle/book/santander')

import pandas as pd
import numpy as np
import xgboost as xgb

# 데이터를 불러온다.
trn = pd.read_csv('./train_ver2.csv', skiprows=range(15000,13647306))
tst = pd.read_csv('./test_ver2.csv', skiprows=range(15000,13647306))

## 데이터 전처리 ##
# 제품 변수를 별도로 저장해 놓는다.
prod = trn.columns[24:].tolist()
# 제품 변수 결측값을 미리 0으로 대체.
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)
# 24개 제품 중 하나도 보유하지 않은 고객 데이터를 제거한다.
no_product = trn[prods].sum(axis=1)==0
trb = trn[~no_product]
# 훈련 데이터와 테스트 데이터를 통합한다.
 # (테스트 데이터에 없는 제품 변수는 0으로 채운다.)
for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn,tst], axis = 0)
# 학습에 사용할 변수를 담는 list.
features = []
# 범주형 변수를 .factorize() 함수를 통해 label encoding한다.
categorical_cols =['ind_empleado','pais_residencia','sexo','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall','tipodom','nomprov','segmento']

for col in categorical_cols:
    df[col], _= df[col].factorize(na_sentinel=-99) # na_sentinel, factorize에 포함되지 않는 변수는 다른 값으로 대체
features += categorical_cols

# 수치형 변수의 특이값과 결측값을 -99로 대체하고, 정수형으로 변환한다.
df['age'].replace(' NA', -99, inplace=True)
df['age'] = df['age'].astype(np.int8)
              
df['antiguedad'].replace('     NA',-99, inplace = True)    
df['antiguedad'] = df['antiguedad'].astype(np.int8)

df['renta'].replace('         NA',-99, inplace = True)
df['renta'].fillna(-99, inplace = True)
df['renta'] = df['renta'].astype(float).astype(np.int8)

df['indrel_1mes'].replace('P',5,inplace=True)
df['indrel_1mes'].fillna(-99,inplace=True)
df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)

# 학습에 사용할 수치형 변수를 feature에 추구한다.
features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']


# =============================================================================
#  2) 피처 엔지니어링
# =============================================================================
피처 엔지니어링 단계에서는 머신러닝 모델 학습에 사용할 파생 변수를 생성한다.
Baseline 모델에서는 전체 24개의 고객 변수와, 4개의 날짜 변수 기반 파생 변수 그리고 24개의 lag-1 변수를 사용한다.

# (코드 2-16) Baseline 모델 피처 엔지니어링 코드
# (피처 엔지니어링) 두 날짜 변수에서 연도와 월 정보를 추출한다.
df['fecha_alta_month'] = df['fecha_alta'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['fecha_alta_month','fecha_alta_year']

df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
# 그 외 변수의 결측값은 모두 -99로 대체한다.
df.fillna(-99, inplace=True)
# (피처 엔지니어링) lag-1 데이터를 생성한다.
# 코드 2-12와 유사한 흐름이다.

# 날짜를 숫자로 변환하는 함수이다.2015-01-28은 1, 2016-05-28은 18로 변환한다.
def date_to_int(str_date):
    Y,M,D = [int(a) for a in str_date.strip().split('-')]
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date
# 날짜를 숫자로 변환해 int_date에 저장한다.
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# 데이터를 복사하고 int_date 날짜에 1을 더해 lag를 생성한다.(변수명에 _prev 추가)
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers','int_date'] else col for col in df.columns]
df_lag['int_date'] += 1

# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다.
# (lag 데이터의 int_date는 1 밀려 있기 때문에, 저번 달의 제품 정보가 삽입된다.)
df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how = 'left')

# 메모리 효율 위해 불필요한 변수를 메모리에서 제거
del df, df_lag

# 저번 달의 제품 정보가 존재하지 않을 경우를 대비해 0으로 대체한다.
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0,inplace=True)
df_trn.fillna(-99, inplace = True)

# lag-1 변수 추가
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]

###
### Baseline 모델 이후, 다양한 피처 엔지니어링을 여기에 추가한다.
###

# =============================================================================
# 3. 머신러닝 모델 학습
# =============================================================================
[교차 검증]
* 내부 교차 검증 과정에서도 최신 데이터(2016-05-28)를 검증 데이터로 분리하고 나머지 데이터를 훈련 데이터로 사용하는 것이 일반적.
* Baseline 모델에서는 모델을 간소화하기 위해 2016-01-28 ~ 2016-04-28로 총 4개월치 데이터를 훈련 데이터 사용하로, 2016-05-28 데이터를 검증 데이터로 사용한다.

# (코드 2-17) 교차 검증을 위해 데이터 분리하기 : 훈련 데이터 전체를 사용하지 않고 2016년도만 사용하도록 추출하는 부분은 피처 엔지니어링에 해당될 수 있다.
## 모델 학습
# 학습을 위해 데이터를 훈련/테스트용으로 분리한다.
# 학습에는 2016-01-28 ~ 2016-04-28 데이터만 사용하고, 검증에는 2016-05-28 데이터를 사용한다.
use_dates = ['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28']
trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]
tst = df_trn[df_trn['fecha_dato'] == '2016-06-28']
del df_trn


# 훈련 데이터에서 신규 구매 건수만 추출한다.
X=[]; Y=[]
for i ,prod in enumerate(prods):
    prev = prod + '_prev'
    prX = trn[(trn[prod]==1) & trn[prev]==0]
    prY = np.zeros(prX.shape[0], dtype = np.int8) + i
    X.append(prX)
    Y.append(prY)
XY = pd.concat(X)
Y = np.hstack(Y)
XY['y'] = Y


# 훈련, 검증 데이터로 분리한다.
vld_date = '2016-05-28'
XY_trn = XY[XY['fecha_dato'] != vld_date]
XY_vld = XY[XY['fecha_dato'] == vld_date]

[ 모델 ]
xgboost
* max_depth, 트리 모델의 최대 깊이
             값이 높을 수록 더 복잡한 트리 모델 생성하며, 과적합의 원인이 될 수 있다.
* eta, 딥러닝에서의 learning rate와 같은 개념이다.
       0과 1 사이의 값을 가지며, 값이 너무 높으면 학습이 잘 되지 않을 수 있으며, 
       반대로 값이 너무 낮을 경우 학습이 느릴 수 있다.
* colsample_bytree, 트리를 생성할 때 훈련 데이터에서 변수 샘플링해주는 비율
                    모든 트리는 전체 변수의 일부만을 학습해 서로의 약점을 보완한다. (0.6 - 0.9 사이의 값을 사용)
* colsample_bylevel, 트리의 레벨 별로 훈련 데이터의 변수를 샘플링해주는 비율. (0.6 - 0.9 사이의 값 사용)
=> 시간 투자 대비 효율을 생각한다면 파라미터 튜닝보다 피처 엔지니어링에 더 많은 시간을 쏟을 것을 권장!!

# (코드 2-18) XGBoost 모델을 훈련 데이터에 학습하는 코드
# XGBoost 모델 파라미터 설정
param = { 
    'booster' : 'gbtree',
    'max_depth': 8,
    'nthread' : 4,
    'num_class' : len(prods),
    'objective' : 'multi:softprob',
    'silent':1,
    'eval_metric':'mlogloss',
    'eta':0.1,
    'min_child_weight':10,
    'colsapole_bytree':0.8,
    'colsample_bylevel':0.9,
    'seed':2018
    }
# 훈련, 검증 데이터를 XGBoost 형태로 변환한다.

XY_trn[features].values
X_trn = XY_trn[features].values
Y_trn = XY_trn['y'].values
dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names = features)

X_vld = XY_vld[features].values
Y_vld = XY_vld['y'].values
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names = features)

# XGBoost 모델을 훈련 데이터로 학습한다.
watch_list = [(dtrn, 'train'), (dvld, 'eval')]
model = xgb.train(param, dtrn, num_boost_round = 1000, evals = watch_list, early_stopping_rounds = 20)

# 학습한 모델을 저장한다.
import pickle
pickle.dump(model, open("model/xgb.baseline.pkl",'wb'))
best_ntree_limit = model.best_ntree_limit

# (코드 2-19) 검증 데이터에 대한 MAP@7 값을 구하는 코드
# AMP@7 평가 척도를 위한 준비 작업니다.
# 고객 식별 번호를 추출한다.
vld = trn[trn['fecha_dato'] == vld_date]
ncodpers_vld = cld['ncodpers'].values
# 검증 데이터엥서 신규 구매를 한다.
for prod in prods:
    prev = prod + '_prev'
    padd = prod _ '_add'
    vld[padd] = vld[prod] - vld[prev]
add_vld = vld[prod + 'add' for prod in prods]
add_vld_list = [list() for i in range(len(ncodpers_vld))]

# 고객별 신규 구매 정답값을 add_vld_list에 저장하고, 총 count를 count_vld에 저장한다.
count_vld = 0
for ncodper in range(len(ncodpers_vld)):
    for prod in range(len(prods)):
        if add_vld[ncodper, prod] > 0:
            add_vld_list[ncodper].append(prod)
            count_vld += 1
        
# 검증 데이터에서 얻을 수 있는 MAP@7 최고점을 미리 구한다. (0.042663)
print(mapk(add_vld_list, add_vld_list, 7, 0.0))

# 검증 데이터에 대한 예측 값을 수한다.
X_vld = vld[features].values
Y_vld = vld['y'].values
dvld = xgb.DMatrix(X_vld, label = Y_vld, feature_names = features)
pred_vld(dvldm ntree_limit = best_ntree_limit)

# 저번 달에 보유한 제품은 신규 구매가 불가하기 때문에, 확률값에서 미리 1을 빼준다.
preds_vld = preds_vld - vld[prod + '_prev' for prod in prods]
# 검증 데이터 예측 상위 7개를 추출한다.
result_vld = []
for ncodper, pred in zip(ncodpers_vld, preds_vld):
    y_prods = [(y,p,ip) for y,p,ip zip(pred, prods,range(len(prods)))]
    y_prods = sorted(y_prods, key - lambda a: a[0], reverse=True)[:7]
    
# 검증 데이터에서의 MAP@7 점수를 구한다.
print(mapk(add_vld_list, result_vld,7,0.0))

=>Baseline 모델은 검증 데이터에서 MAP@7 0.36466점을 기록한다.
검증데이터 최고 점수가 0.042663임을 감안하면 Baseline 모델의 정확도는 (0.036466/0.042663)=0.85
약, 85% 수준이다.
이 정확도 수준의 높낮이 알기위해서는 캐글에 직접 제출

# =============================================================================
# 4. 테스트 데이터 예측 및 캐글 업로드
# =============================================================================
테스트 데이터에 대해 조금이라도 좋은 성능을 내기 위해, 훈련 데이터와 검증 데이터를 합친 전체 데이터에 대하여 XGBoost모델 재학습
XGBoost 모델의 파라미터는 교차 검증을 통해 찾아낸 최적의 파라미터를 사용하되, 
XGBoost 모델에 사용되는 트리의 개수르 늘어난 검증 데이터만큼 증가한다.

모델의 변수 중요도 출력 -> get_fscore()

# (코드2-20) Baseline 모델 캐글 제출용 파일 생성하는 코드
# XGBoost 모델을 전체 훈련 데이터로 재학습한다.
x_all = XY[features].values
y_all = XY['y'].values
dall = xgb.DMatrix(X_all, label = Y_all , feature_names = features)
watch_list = [(dall,'train')]
# 트리 개수를 늘어난 데이터 양만큼 비례해서 증가한다.
best_ntree_limit = int(best_ntree_limit * (len(XY_trn) + len(xy_vld)) / len(XY_trn))
# XGBoost 모델 재학습
model = xgb.train(param, dall, num_boost_round = best_ntree_limit, evals = wahtc_list)
# 변수 중요도 출력한다. 예상하던 변수가 상위에 있는지 여부 확인
print('Feature importance : ')
for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv : kv[1], reverse=True):
    print(kv)
# 캐글 제출 위해 테스트 데이터에 대한 예측값 출력
x_tst - tst[features].values
dtst = xgb.DMartix(X_tst, feature_names=features)
pred_tst = model.predict(dtst, ntree_limit = best_ntree_limit)
ncodpers_tst = tst['ncodpers'].values
preds_tst = preds_tst - tst[prod + '_prev' for prod in prods]

# 제출 파일 생성
submit_file = open('./xgb.baseline.2020.11.','w')
submit_file.write('ncodpers,added_products\n')
for ncodper, pred in zip(ncodpers_tst, preds_tst):
    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods,range(len(prods)))]
    y_prods = sorted(y_prods, key=lambda a : a[0], reverse = True)[:7]
    y_prods = [p for y,p,ip un y_prods]
    submit_file.write('{},{}\n'.format(int(ncodper),' '.join(y_prods)))
    


