# =============================================================================
# (Kaggle) Bike Sharing Demand
# =============================================================================
# https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile
# path
import os
os.getcwd()
os.chdir('./.spyder-py3/kaggle/bike')

# os.path.dirname(os.path.abspath('__Bike.py__'))  # 절대경로


# import library
import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category = DeprecationWarning)

# =============================================================================
# Read In the Dataset
# =============================================================================
dailyDate = pd.read_csv('train.csv')

# 데이터에 대해 첫번째 단계(3가지)
# 1) 데이터 크기 확인
# 2) 일부 데이터 확인
# 3) 데이터 변수 유형 확인

dailyDate.shape  # (10886, 12)
dailyDate.head()
dailyDate.info()


# Feature Engineering
# 일부 열 범주화(season, holiday, workingday, weather)
#  - 'datetime'컬럼에서 'date', 'hour','weekDay','month' 컬럼 생성
#  - 'season', 'holiday', 'workingday' 범주화
#  - 위 컬럼 추출 후 datetime 컬럼 삭제
 
# Creating New Columns From 'Datatime' Columns
dailyDate.columns
dailyDate['date'] = dailyDate.datetime.apply(lambda x : x.split()[0])
dailyDate['hour'] = dailyDate.datetime.apply(lambda x : x.split()[1].split(':')[0])
dailyDate['weekday']  = dailyDate.date.apply(lambda datestring : calendar.day_name[datetime.strptime(datestring,'%Y-%m-%d').weekday()])
dailyDate['month'] = dailyDate.date.apply(lambda datestring : calendar.month_name[datetime.strptime(datestring,'%Y-%m-%d').month])
dailyDate['season'] = dailyDate.season.map({1:'Spring', 2:'Summer',3:'Fall',4:'Winter'})
dailyDate['weather']= dailyDate.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
                                             2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
                                             3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
                                             4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })

# 컬럼 범주화
categoryVariableList = ['hour','weekday','month','season','weather','holiday','workingday']

for var in categoryVariableList :
    dailyDate[var] = dailyDate[var].astype('category')


# 불필요한 컬럼 제거 (datetime 컬럼 제거)
dailyDate = dailyDate.drop(['datetime'],axis = 1)

# =============================================================================
# Lets Start With Very Simple Visualization Of Variables DataType Count
# 데이터 타입 시각화
# =============================================================================
dataTypeDf = pd.DataFrame(dailyDate.dtypes.value_counts()).reset_index().rename(columns={'index':'variableType', 0:'count'})

fig, ax = plt.subplots()
fig.set_size_inches(12,5)

sns.barplot(data=dataTypeDf,x='variableType',y='count',ax=ax)
ax.set(xlabel = 'variableTypeariable Type', ylabel = 'Count', title = 'Variables DataType Count')

# =============================================================================
# Missing Values Analysis 
# =============================================================================
msno.matrix(dailyDate, figsize = (12,5))
# ==> 결측치 없음

# =============================================================================
# Outliers Analysis
# =============================================================================
fig, axes = plt.subplots(nrows = 2, ncols = 2)
fig.set_size_inches(12,10)

sns.boxplot(data=dailyDate, y='count', orient='v', ax=axes[0][0])  # orinet = 'v', vertical(세로방향)
sns.boxplot(data=dailyDate, y='count',x='season', orient='v',ax=axes[0][1])
sns.boxplot(data=dailyDate, y='count',x='hour', orient='v', ax=axes[1][0])
sns.boxplot(data=dailyDate, y='count',x='workingday', orient='v', ax=axes[1][1])

axes[0][0].set(ylabel='Count', title='Box Plots On Count')
axes[0][1].set(xlabel='Season', title='Box Plot On Count Across Hour Of The Day')
axes[1][0].set(xlabel='hour Of The Day', ylabel='Count', title='Box polt On Count Across Hour Of The Day')
axes[1][1].set(xlabel='Working Day', ylabel='Count', title='Box Plot On Xount Across Working Day')

# =============================================================================
# Remove Outliers In The Count Column
# =============================================================================
dailyDateWithoutOutliers = dailyDate[np.abs(dailyDate['count']-dailyDate['count'].mean())<=(3*dailyDate['count'].std())]

print('Shape Of The Before Outliers :',dailyDate.shape)
print('Shape Of The Before Oltliers :',dailyDateWithoutliers.shape)

# =============================================================================
# Correlation Analysis
# =============================================================================

 - 종속변수가 변수에 의해 어떻게 영향을 받는지 이해하는 일반적 방법은 변수 사이 상관행렬을 섬유화하는 것이다.
1) 온도와 습도 변수는 각각 양과 음의 상관관계가 있다. 이 둘의 상관 관계는 매우 두드러지진 않지만, 수치 변수는 'temp'와'humidity'에 거의 의존하지 않는다.
2) 'windspeed'는 실제 유용한 수치 변수가 아니며 'count'와의 상관 관계 값에서 볼 수 있다.
3) 'temp'와 'atemp'가 서로 강한 상관 관계를 가지고 있어 'atemp'는 고려되지 않는다.
    모델 구축시 변수가 데이터에서 다중 공선성을 나타내므로 변수 중 하나를 삭제해야 한다.
4) 'casual'과 'registered' 또한 본질적으로 누수되는 변수이므로 모델 구축 중 삭제해야하므로 고려하지 않는다.

# 'count' vs 'temp','humidity','windspeed' 고려

corrMatt = dailyDate[['temp','atemp','casual','registered','humidity','windspeed','count']].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False

fig,ax = plt.subplots()
fig.set_size_inches(10,5)
sns.heatmap(corrMatt, mask = mask, vmax = .8, square = True, annot = True)

fig, (ax1,ax2,ax3) = plt.subplots(ncols = 3)
fig.set_size_inches(10,5)
sns.regplot(x = 'temp', y = 'count', data = dailyDate, ax = ax1)
sns.regplot(x = 'windspeed', y='count', data = dailyDate, ax = ax2, color = 'r')
sns.regplot(x = 'humidity', y = 'count', data = dailyDate, ax = ax3, color = 'g')


# =============================================================================
# Visualizing Distribution Of Data
# =============================================================================
아래 그림에서 볼 수 있듯이 "count" 변수가 오른쪽으로 치우쳐 있다.
기계학습기술의 대부분은 종속변수가 정상이어야 하므로 정규분포를 따르는 것이 바람직하다.
한 가지 가능한 해결책은 특이치 데이터 점을 제거한 후 "count" 변수에 대한 로그 변환을 수행하는 것이다. 
변환 후 데이터는 훨씬 좋아 보이지만 여전히 이상적으로 정규 분포를 따르지는 않는다.

fig, axes = plt.subplots(ncols = 2, nrows = 2)
fig.set_size_inches(12,10)
sns.distplot(dailyDate['count'], ax = axes[0][0])
stats.probplot(dailyDate['count'], dist='norm', fit = True, plot = axes[0][1])
sns.distplot(np.log(dailyDateWithoutOutliers['count']), ax=axes[1][0])
stats.probplot(np.log1p(dailyDateWithoutOutliers['count']), dist = 'norm', fit = True, plot = axes[1][1])

# =============================================================================
# Visualizing Count Vs (Month, Seadon, Hour, Weekday, Usertype)
# =============================================================================
1) 사람들이 여름동안 자전거를 빌리는 경향이 있는데, 이는 자전거를 타기 좋은 계절이기 때문이다.
2) 평일 7-8AM, 5-6PM 사이 사람들이 자전거를 더 빌리는 사람들이 많아지는 경향이 있다.
   앞에 언급했듯 이것은 일반적인 학생과, 근로자에 기인한다.
3) 위의 패턴은 주말(토,일)에는 관찰되지 않는다. 주말에는 10AM - 4PM 사이 자전거를 빌리는 경향이 있다.
4) 최대 사용자 수는 7-8 am, 5-6 pm 의 등록된 사용자가 원인이 된다.

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4)
fig.set_size_inches(12,20)

sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

monthAggregated = pd.DataFrame(dailyDate.groupby('month')['count'].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by = 'count', ascending = False)
sns.barplot(data = monthSorted, x = 'month', y = 'count', ax = ax1, order=sortOrder)
ax1.set(xlabel='Month', ylabel='Average Count', title='Average Count By Month')

hourAggregated = pd.DataFrame(dailyDate.groupby(['hour','season'],sort=True)['count'].mean()).reset_index()
sns.pointplot(x=hourAggregated['hour'], y=hourAggregated['count'], hue=hourAggregated['season'],data=hourAggregated, join=True, ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='User Count', title='Average User Count By Hour Of The Day Across Season', label='big')

hourAggregated = pd.DataFrame(dailyDate.groupby(['hour','weekday'],sort=True)['count'].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax3)
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')

hourTransForm = 
# =============================================================================
# 부록
# =============================================================================
import pylab
 - numpy와 matplotlib.pyplot 모듈을 불러오는 방법
 - pylab을 부르면 numpy를 import하지 않아도 되나 namespace를 같이 쓰는 것이 문제 될 수 있으므로 위처럼 따로 불러 사용한다.
 
import calendar
 - 달력 출력하는 모듈.(월요일을 처음, 일요일을 마지막)
 https://python.flowdas.com/library/calendar.html (참고)
 
sns.boxplot(kwargs)
https://buillee.tistory.com/198
 
np.triu_indices_from(arr)
 - 삼각형에 대한 지수를 반환
 https://numpy.org/doc/stable/reference/generated/numpy.triu_indices_from.html
 
sns.heatmap(df, # 데이터
            vmin = 100, # 최솟값
            vman = 700, # 최댓값
            cbar = True, # colorbar 유무
            center = 400, # 중앙값 선정
            linewidths = 0.5,   # cell 사이에 선 표
            annot = True, # 각 cell의 값 표기 유무
            fmt = 'd' # 그 값의 데이터 타입 설정
            cmap = 'Blues' # 히트맵 색 설정
            )
https://dsbook.tistory.com/51
