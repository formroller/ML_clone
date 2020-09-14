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

# =============================================================================
# Outliers Analysis
# =============================================================================

# =============================================================================
# 부록
# =============================================================================
import pylab
 - numpy와 matplotlib.pyplot 모듈을 불러오는 방법
 - pylab을 부르면 numpy를 import하지 않아도 되나 namespace를 같이 쓰는 것이 문제 될 수 있으므로 위처럼 따로 불러 사용한다.
 
import calendar
 - 달력 출력하는 모듈.(월요일을 처음, 일요일을 마지막)
 https://python.flowdas.com/library/calendar.html (참고)
 
