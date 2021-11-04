import os
os.getcwd()
os.chdir('./.spyder-py3/wine')  # 'C:\\Users\\yongjun\\.spyder-py3\\wine'

# 1. 라이브러리 호출
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm

# 2. 데이터 로딩
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
test.head()

train.info()
test.info()
# 결측치 없음, test에 quaility 변수 없음

# train 변수간 상관관계 분석
plt.figure(figsize=(12,12))
sns.heatmap(data = train.corr(), annot = True, fmt = '.2f', linewidths = .5, cmap = 'Blues')
# (ann = True(annotate each cell with numeric value) 각 셀에 숫자 입력)

# test 변수간 상관관계 분석
plt.figure(figsize=(12,12))
sns.heatmap(data = test.corr(), annot = True, fmt = '.2f', linewidths = .5, cmap = 'Reds')

# train 변수별 분포 확인
plt.figure(figsize = (12,12))
for i in range(1,13):
    plt.subplot(3,4,i)
    sns.distplot(train.iloc[:,i])
plt.tight_layout()
plt.show()


# test 변수별 분포 확인
plt.figure(figsize = (12,12))
for i in range(1,12):
    plt.subplot(3,4,i)
    sns.distplot(test.iloc[:,i])
plt.tight_layout()    
plt.show()


# train에서 각 변수와 quality 변수 사이 분포 확인
for i in range(11):
    fig = plt.figure(figsize=(12,6))
    sns.barplot(x = 'quality', y = train.columns[i+2], data = train)

# 3. 변수 변환
train['type'] = train['type'].map({'white':0,'red':1}).astype('int')
test['type'] = test['type'].map({'white':0,'red':1}).astype('int')
# pd.get_dummies(train['type'])


# 모델 입력 전 데이터 정형화
train_x = train.drop(['index','quality'], axis = 1)
train_y = train['quality']
test_x = test.drop('index', axis = 1)

train_x.shape, train_y.shape, test_x.shape

# 4. 모델 생성 및 훈련
model = lgbm.LGBMClassifier()
model.fit(train_x, train_y)

# 5.생성한 모델로 예측 데이터 생성
y_pred = model.predict(test_x)
y_pred

# 6. 제출파일
sm_sub = pd.read_csv('sample_submission.csv')
sm_sub['quality'] = y_pred

sm_sub.to_csv('submission.csv', index = False)
