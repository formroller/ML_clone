import os
os.getcwd()
os.chdir('C:/Users/yongjun/.spyder-py3/kaggle/titanic')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category = DeprecationWarning)

Titanic = pd.read_csv('train.csv')

Titanic.shape
Titanic.info()
Titanic.head()
msno.matrix(Titanic, figsize=(12,5))

sns.heatmap(data = Titanic, x)

# Feature Engineering
# 일부 열 범주화(Survived, Pclass, Sex , sibSp, parch, Fare)
#  - 'datetime'컬럼에서 'date', 'hour','weekDay','month' 컬럼 생성
#  - 'season', 'holiday', 'workingday' 범주화
#  - 위 컬럼 추출 후 datetime 컬럼 삭제
 
# Creating New Columns From 'Datatime' Columns





