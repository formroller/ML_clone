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


 - 종속변수가 변수에 의해 어떻게 영향을 받는지 이해하는 일반적 방법은 변수 사이 상관행렬을 섬유화하는 것이다.
1) 온도와 습도 변수는 각각 양과 음의 상관관계가 있다. 이 둘의 상관 관계는 매우 두드러지진 않지만, 수치 변수는 'temp'와'humidity'에 거의 의존하지 않는다.
2) 'windspeed'는 실제 유용한 수치 변수가 아니며 'count'와의 상관 관계 값에서 볼 수 있다.
3) 

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
