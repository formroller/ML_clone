# Kaggle - Wine Quality Prediction
import os
os.getcwd()
os.chdir('./wine')
# library import 
import pandas as pd
import seaborn as sbs
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# 1. loading dataset
wine = pd.read_csv('redwine.csv')

wine.head()
wine.info()

# =============================================================================
# ## ML을 위한 데이터 전처리
# =============================================================================

# 2. quality별 상관관계
for i in range(len(wine.columns)):
    fig = plt.figure(figsize=(7,6))
    sns.barplot(x = 'quality', y = wine.columns[i-1], data = wine)

# 이진분류 (좋음/나쁨)
bins = (2, 6.5, 8) # 미포함, 포함
group_names = ['bad','good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

# 품질 변수 범주화
label_qulity = LabelEncoder()

# Bad becomes 0 and good becomes 1
wine['quality'] = label_qulity.fit_transform(wine['quality'])
wine['quality'].value_counts()

sns.countplot(wine['quality'])

# 데이터 분리
X = wine.drop('quality', axis = 1)
y = wine['quality']

# Train / Test set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 최적화 값 얻기 위해 표준화 적용
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# =============================================================================
#  ML을 위한 train / test data 준비 완료
# =============================================================================

# 1) Random Forest
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

# 모델 작동 결과 확인
print(classification_report(y_test, pred_rfc))

# Confusion Matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))
 # ==> Random Forest 정확도 : 89%

# 2) 경사하강 분류 (SGDClassifier)
sgd = SGDClassifier(penalty = 'none')
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))

# ==> SGDClassifer 정확도 : 85%


# 3-1) Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))

# ==> SVC 정확도 : 88% (grid search 전)

# (모델 정확도 향상)
# Grid Search CV

# finding best parameters for our SVC model
param = {
    'C' : [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
    'kernel' : ['linear', 'rbf'],
    'gamma' : [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
    }

grid_cv = GridSearchCV(svc, param_grid = param, scoring = 'accuracy', cv = 10)
grid_cv.fit(X_train, y_train)
# best parameters for our svc model
grid_cv.best_params_  # {'C': 1.2, 'gamma': 0.9, 'kernel': 'rbf'}

# 3-2) SVC (best params)
svc2 = SVC(C = 1.2, gamma = 0.9, kernel = 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
# ==> SVC@ (best params) 정확도 : 90%
# => 2% 향상


# Cross Validation Score for random forest 
 # - 교차검증(Cross Validation)을 통한 RF모델 평가
rfc_eval = cross_val_score(rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()
# ==> 0.913238188976378
# 정확도 3% 향상.


# =============================================================================
# ## 용어 설명
# =============================================================================
Confusion Matrix(오차행렬)
 - training을 통한 prediction 성능을 측정하기 위해 예측 value와 실제 value를 비교하기 위한 표
 - 분류에 한정되어 사용
 
              precision    recall  f1-score   support

           0       0.88      0.96      0.91       273
           1       0.45      0.21      0.29        47

    accuracy                           0.85       320
   macro avg       0.67      0.58      0.60       320
weighted avg       0.81      0.85      0.82       320

precision 0.88 -> 결과에서 0으로 예측한 데이터의 88%가 실제 0
recall 0.96 -> 실제 0인 데이터 중 96% 0으로 예측

macro : 단순평균
weighted : 각 클래스에 속하는 표본의 갯수로 가중평균
accuracy : 정확도, 전체 학습 데이터 갯수 중 각 클래스에서 자신의 클래스를 정확히 맞춘 개수의 비율

