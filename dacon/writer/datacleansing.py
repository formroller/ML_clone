import os 
os.chdir('./kaggle/dacon_writer')
# =============================================================================
# 2. Data Cleansing & Pre-Processing
# =============================================================================
import pandas as pd 
import numpy as np

import re
# nltk?
import nltk
import nltk.data
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn import metrics, preprocessing, pipeline, model_selection, naive_bayes
from sklearn.metrics import log_loss  #?
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline #?
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer #?
from sklearn.naive_bayes import MultinomialNB, BernoulliNB #?
from sklearn.calibration import CalibratedClassifierCV #?
from sklearn.linear_model import SGDClassifier, LogisticRegression
import xgboost as xgb

import time

# keras
from keras import backend as K #?
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence, text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

## 데이터 전처리
pd.set_option('display.max_columns',200)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test_x.csv')


X_train = train['text'].str.replace('[^a-zA-Z0-9]',' ') # train['text'] 특수기호 제거
Y_train = LabelEncoder().fit_transform(train['author']) # train['author'] 범주화
y_train = train['author'] # target 변수에 할당
X_test = test['text'].str.replace('^[a-zA-Z0-9]',' ')  # test['text'] 특수기호 제거


# 1) python dict in list
punctuations=[{"id":1,"p":"[;:]"},   
              {"id":2,"p":"[,.]"},
              {"id":3,"p":"[?]"},
              {"id":4,"p":"[!]"},
              {"id":5,"p":"[''\']"},
              {"id":6,"p":"[""\"]"},
              {"id":7,"p":"[:;,.?! \' "" '' \"]"}]

for p in punctuations:
    punctuation = p['p']
    _train = [sentence.split() for sentence in train['text']]
    train['punc_' + str(p['id'])] = [len([word for word in sentence if bool(re.search(punctuation, word))]) * 100 / len(sentence) for sentence in _train] 
        
    _test = [sentence.split() for sentence in test['text']]
    test['punc_' + str(p['id'])] = [len([word for word in sentence if bool(re.search(punctuation, word))]) * 100 / len(sentence) for sentence in _test]
#=>  문장내 punctuation 기호 일치하는 비율을 찾는다.

## Pipeline
# TfidfVectorizer
# CountVectorizer

## 1.
start = time.localtime()
print('%04d%02d%02d%02d:%02d' % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour,start.tm_min)) # 시작시간

# tfidf_MNB_
cv_scores=[] # CrossValidation 담을 list
pred_full_test=0
pred_train = np.zeros([train.shape[0],5]) # train.shape[0],5 = train.shape[0]*5 크기 값이 0인 배열 생성

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 32143233) 
for dev_index, val_index in kf.split(train): 
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    
    # Pipeline
    classifier = Pipeline([('vect', TfidfVectorizer(lowercase=False)),
                           ('tfidf',TfidfTransformer()),
                           ('clf',MultinomialNB()),
                          ])
    
    # parameter 설정, Pipeline시 지정할문자열__하이퍼파라미터 문자열
    parameters = {'vect__ngram_range':[(1,2)],
                  'vect__max_df':(0.25,0.3),
#                   'vect__min_df':[1],
                  'vect__analyzer':['word'],
                  'clf__alpha':[0.024, 0.031],
                 }
    
    gs_clf = GridSearchCV(classifier, parameters, n_jobs =-1, verbose=1, cv=2) # Pipeline 적용한(classifier,parameter) GridsearchCV 할당
    gs_clf.fit(dev_X, dev_y) 
    best_parameters = gs_clf.best_estimator_.get_params() 
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r'%(param_name, best_parameters[param_name]))  # %r, 문자열
    
    #2) Predict_proba
    pred_test_y = gs_clf.predict_proba(val_X) # val_X 클래스에 대한 예측 확률.
    pred_test_y2 = gs_clf.predict_proba(X_test)
    pred_full_test = pred_full_test + pred_test_y2
    pred_train[val_index, : ] = pred_test_y  
    cv_scores.append(metrics.log_loss(val_y, pred_test_y))
        
print('cv socres:',cv_scores)
print('Mean cv score',np.mean(cv_scores))
pred_full_test = pred_full_test/5

train['tfidf_MNB_0'] = pred_train[:,0]
train['tfidf_MNB_1'] = pred_train[:,1]
train['tfidf_MNB_2'] = pred_train[:,2]
train['tfidf_MNB_3'] = pred_train[:,3]
train['tfidf_MNB_4'] = pred_train[:,4]

test['tfidf_MNB_0'] = pred_full_test[:,0]
test['tfidf_MNB_1'] = pred_full_test[:,1]
test['tfidf_MNB_2'] = pred_full_test[:,2]
test['tfidf_MNB_3'] = pred_full_test[:,3]
test['tfidf_MNB_4'] = pred_full_test[:,4]

end = time.localtime()

print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))



## 2
# Pipeline - clf 변경
start = time.localtime()
print('%04d%02d%02d%02d:%02d' % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour,start.tm_min))

# tfidf_MNB_
cv_scores=[]
pred_full_test=0
pred_train = np.zeros([train.shape[0],5])

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 32143233)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    
    # 위와 clf 부분 다름
    classifier = Pipeline([('vect', TfidfVectorizer(lowercase=False)),
                          ('tfidf',TfidfTransformer()),
                          ('clf',CalibratedClassifierCV(MultinomialNB(alpha = 0.05), method = 'isotonic')), 
                          ])
    # clf_alpha 주석
    parameters = {'vect__ngram_range':[(1,2)],
                 'vect__max_df': (0.4, 0.5),
                 #'vect__min_df':[1],
                  'vect__analyzer':['word'],
                 #'clf__alpha' :(0.016, 0.018),
                 }

    
    gs_clf = GridSearchCV(classifier, parameters, n_jobs = -1, verbose=1, cv=2)
    gs_clf.fit(dev_X, dev_y)
    best_parameters = gs_clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' %(param_name, best_parameters[param_name]))
            
    pred_test_y = gs_clf.predict_proba(val_X)
    pred_test_y2 = gs_clf.predict_proba(X_test)
    pred_full_test = pred_full_test + pred_test_y2
    pred_train[val_index,:] = pred_test_y
    cv_scores.append(metrics.log_loss(val_y, pred_test_y))
    
print('cv socre:',cv_scores)
print('Mean cv socre:', np.mean(cv_scores))
pred_full_test = pred_full_test/5
    
train['tfidf_MNB_0'] = pred_train[:,0]
train['tfidf_MNB_1'] = pred_train[:,1]
train['tfidf_MNB_2'] = pred_train[:,2]
train['tfidf_MNB_3'] = pred_train[:,3]
train['tfidf_MNB_4'] = pred_train[:,4]

test['tfidf_MNB_0'] = pred_full_test[:,0]
test['tfidf_MNB_1'] = pred_full_test[:,1]
test['tfidf_MNB_2'] = pred_full_test[:,2]
test['tfidf_MNB_3'] = pred_full_test[:,3]
test['tfidf_MNB_4'] = pred_full_test[:,4]


## 3 clf, MultinomialNB -> BernoulliNB
start = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
# tfidf_CBNB_
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 5])

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 32143233)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]

    classifier = Pipeline([('vect', TfidfVectorizer(lowercase=False)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', CalibratedClassifierCV(BernoulliNB(alpha = 0.02), method='isotonic')), # 3.calibration
    ])
    parameters = {'vect__ngram_range': [(1, 2)],
                  'vect__max_df': (0.03, 0.4),
#                   'vect__min_df': [1],
                  'vect__analyzer' : ['word'],
#                   'clf__alpha': (0.016, 0.018),
    }
    gs_clf = GridSearchCV(classifier, parameters, n_jobs=-1, verbose=1, cv=2)
    gs_clf.fit(dev_X, dev_y)
    best_parameters = gs_clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    pred_test_y = gs_clf.predict_proba(val_X)
    pred_test_y2 = gs_clf.predict_proba(X_test)
    pred_full_test = pred_full_test + pred_test_y2
    pred_train[val_index, : ] = pred_test_y
    cv_scores.append(metrics.log_loss(val_y, pred_test_y))
print("cv score : ", cv_scores)
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5

train["tfidf_CBNB_0"] = pred_train[ : , 0]
train["tfidf_CBNB_1"] = pred_train[ : , 1]
train["tfidf_CBNB_2"] = pred_train[ : , 2]
train["tfidf_CBNB_3"] = pred_train[ : , 3]
train["tfidf_CBNB_4"] = pred_train[ : , 4]
test["tfidf_CBNB_0"] = pred_full_test[ : , 0]
test["tfidf_CBNB_1"] = pred_full_test[ : , 1]
test["tfidf_CBNB_2"] = pred_full_test[ : , 2]
test["tfidf_CBNB_3"] = pred_full_test[ : , 3]
test["tfidf_CBNB_4"] = pred_full_test[ : , 4]    

end = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))

end = time.localtime()

print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))


## 4
# clf, SGDClassifier(loss = 'modified_huber', alpha = 0.00001, max_iter = 10000, tol=1e-4),method='sigmoid'

start = time.localtime()
print('%04d/%02d/%02d/%02d/%02d' % (start.tm_year, start.tm_mon,start.tm_mday, start.tm_hour, start.tm_min))
# stidf_CBNB_
cv_csores =[]
pred_ful_test = 0
pred_train = np.zeros([train.shape[0],5])

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 32143233)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    
    classifier = Pipeline([('vect', TfidfVectorizer(lowercase = False)),
                         ('tfidf',TfidfTransformer()),
                         ('clf',CalibratedClassifierCV(SGDClassifier(loss = 'modified_huber', alpha = 0.00001, max_iter = 10000, tol=1e-4),method='sigmoid')),
#                            ('clf', CalibratedClassifierCV(SGDClassifier(loss='modified_huber', alpha=0.00001, max_iter=10000, tol=1e-4), method='sigmoid')),
                           ])
    parameters = {'vect__ngram_range':[(1,2)],
#                  'vect__max_df':(0/03, 0.4),
#                   'vect__min_df':[1],
                  'vect__analyzer':['word'],
#                   'clf__alpha':(0.016,0.018)
                 }
    
    gs_clf = GridSearchCV(classifier, parameters, n_jobs = -1, verbose = 1, cv =2)
    gs_clf.fit(dev_X,dev_y)
    best_parameters = gs_clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r'%(param_name, best_parameters[param_name]))
        
    pred_test_y = gs_clf.predict_proba(val_X)
    pred_test_y2 = gs_clf.predict_proba(X_test)
    pred_full_test = pred_full_test + pred_test_y2
    pred_train[val_index, :] = pred_test_y
    cv_scores.append(metrics.log_loss(val_y, pred_test_y))
    
print('cv score : ', cv_scores)
print('Meanc cv score : ', np.mean(cv_scores))
pred_full_test = pred_full_test/5

train["tfidf_CBNB_0"] = pred_train[ : , 0]
train["tfidf_CBNB_1"] = pred_train[ : , 1]
train["tfidf_CBNB_2"] = pred_train[ : , 2]
train["tfidf_CBNB_3"] = pred_train[ : , 3]
train["tfidf_CBNB_4"] = pred_train[ : , 4]
test["tfidf_CBNB_0"] = pred_full_test[ : , 0]
test["tfidf_CBNB_1"] = pred_full_test[ : , 1]
test["tfidf_CBNB_2"] = pred_full_test[ : , 2]
test["tfidf_CBNB_3"] = pred_full_test[ : , 3]
test["tfidf_CBNB_4"] = pred_full_test[ : , 4]    

end = time.localtime()
print('%04d/%02d/%02d/%02d/%02d' % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print('%04d/%02d/%02d/%02d/%02d' % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))

## 5.
# clf, SGDClassifier -> LogisticsRegression
start = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
# tfidf_L_
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 5])

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 32143233)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]

    classifier = Pipeline([('vect', TfidfVectorizer(lowercase=False)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LogisticRegression(C=50, max_iter=200)),
    ])
    parameters = {'vect__ngram_range': [(1, 2)],
#                   'vect__max_df': (0.3, 0.4),
#                   'vect__min_df': [1],
                  'vect__analyzer' : ['word'],
#                   'clf__alpha': (0.016, 0.018),
    }
    gs_clf = GridSearchCV(classifier, parameters, n_jobs=-1, verbose=1, cv=2)
    gs_clf.fit(dev_X, dev_y)
    best_parameters = gs_clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    pred_test_y = gs_clf.predict_proba(val_X)
    pred_test_y2 = gs_clf.predict_proba(X_test)
    pred_full_test = pred_full_test + pred_test_y2
    pred_train[val_index, : ] = pred_test_y
    cv_scores.append(metrics.log_loss(val_y, pred_test_y))
print("cv score : ", cv_scores)
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5

train["tfidf_L_0"] = pred_train[ : , 0]
train["tfidf_L_1"] = pred_train[ : , 1]
train["tfidf_L_2"] = pred_train[ : , 2]
train["tfidf_L_3"] = pred_train[ : , 3]
train["tfidf_L_4"] = pred_train[ : , 4]
test["tfidf_L_0"] = pred_full_test[ : , 0]
test["tfidf_L_1"] = pred_full_test[ : , 1]
test["tfidf_L_2"] = pred_full_test[ : , 2]
test["tfidf_L_3"] = pred_full_test[ : , 3]
test["tfidf_L_4"] = pred_full_test[ : , 4]    

end = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))

# else
방법1. Pipeline 'vect' -> CountVectorizer 변경 한 후 1 ~ 5 반복
방법2. parameters 'vect__analyzer' -> 'char', 'char_wb' 로 변경.


## Keras

def preprocessFastText(text):
    text = text.replace("' ", " ' ")
    signs = set(';:,.?!\'“”‘’\"')
    prods = set(text) & signs # text 기호들 추출
    if not prods:
        return text # 기호 없는 text return

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )  #4)
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocessFastText(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs

docs = create_docs(train)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= 2])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = max([max(len(l) for l in docs)])

docs = pad_sequences(sequences=docs, maxlen=maxlen)

docs_test = create_docs(test)
docs_test = tokenizer.texts_to_sequences(docs_test)
docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)

xtrain_pad = docs
xtest_pad = docs_test



##
input_dim = np.max(docs) + 1
embedding_dims = 20

def initFastText(embedding_dims,input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

##
start = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))

ytrain_enc = np_utils.to_categorical(Y_train)
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([xtrain_pad.shape[0], 5])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32143233)

for dev_index, val_index in kf.split(xtrain_pad):
    dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
    dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
    
    model = initFastText(embedding_dims,input_dim)
    model.fit(dev_X, dev_y,
              batch_size=32, 
              epochs=40, 
              verbose=1, 
              validation_data=(val_X, val_y),
              callbacks=[earlyStopping])
    
    pred_val_y = model.predict(val_X)
    pred_test_y = model.predict(xtest_pad)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print('')
    print('')    
    print('')    
print("cv score : ", cv_scores)
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5 

train["ff_0"] = pred_train[:,0]
train["ff_1"] = pred_train[:,1]
train["ff_2"] = pred_train[:,2]
train["ff_3"] = pred_train[:,3]
train["ff_4"] = pred_train[:,4]
test["ff_0"] = pred_full_test[:,0]
test["ff_1"] = pred_full_test[:,1]
test["ff_2"] = pred_full_test[:,2]
test["ff_3"] = pred_full_test[:,3]
test["ff_4"] = pred_full_test[:,4]

end = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))


##
max_len = 70
nb_words = 10000

texts_1 = []
for text in train['text']:
    texts_1.append(text)

test_texts_1 = []
for text in test['text']:
    test_texts_1.append(text)

tokenizer = Tokenizer(num_words=nb_words)
tokenizer.fit_on_texts(texts_1)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
word_index = tokenizer.word_index

test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)
xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)
del test_sequences_1
del sequences_1
nb_words_cnt = min(nb_words, len(word_index)) + 1

##

def initNN(nb_words_cnt, max_len):
    model = Sequential()
    model.add(Embedding(nb_words_cnt,32,input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 5, padding='valid', activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

##
start = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))

ytrain_enc = np_utils.to_categorical(Y_train)
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([xtrain_pad.shape[0], 5])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32143233)

for dev_index, val_index in kf.split(xtrain_pad):
    dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
    dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
    
    model = initNN(nb_words_cnt, max_len)
    model.fit(dev_X, dev_y,
              batch_size=32,
              epochs=3,
              verbose=1,
              validation_data=(val_X, val_y),
              callbacks=[earlyStopping])
    
    pred_val_y = model.predict(val_X)
    pred_test_y = model.predict(xtest_pad)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print('')
    print('')
    print('')
print("cv score : ", cv_scores)
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5 

train["nn_0"] = pred_train[:,0]
train["nn_1"] = pred_train[:,1]
train["nn_2"] = pred_train[:,2]
train["nn_3"] = pred_train[:,3]
train["nn_4"] = pred_train[:,4]

test["nn_0"] = pred_full_test[:,0]
test["nn_1"] = pred_full_test[:,1]
test["nn_2"] = pred_full_test[:,2]
test["nn_3"] = pred_full_test[:,3]
test["nn_4"] = pred_full_test[:,4]

end = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))

# =============================================================================
# 모델 학습 및 검증
# =============================================================================
start = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
# Final Model
# XGBoost
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 5
#     param['silent'] = 1
    param['num_class'] = 5
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

def do(train, test, Y_train):
    drop_columns=['index', "text"]
    x_train = train.drop(drop_columns+['author'],axis=1)
    x_test = test.drop(drop_columns,axis=1)
    y_train = Y_train
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32143233)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([x_train.shape[0], 5])
    for dev_index, val_index in kf.split(x_train):
        dev_X, val_X = x_train.loc[dev_index], x_train.loc[val_index]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, x_test, seed_val=0, colsample=0.7)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("cv score : ", cv_scores)
    print("Mean cv score : ", np.mean(cv_scores))
    return pred_full_test/5
result = do(train, test, Y_train)

end = time.localtime()
print("%04d/%02d/%02d %02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min))
print("%04d/%02d/%02d %02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min))

# =============================================================================
# 결과
# =============================================================================
sample_submission=pd.read_csv('open/sample_submission.csv', encoding='utf-8')
sample_submission[['0', '1', '2', '3', '4']] = result
sample_submission.to_csv("sub_4_1210.csv", index=False)
sample_submission

# =============================================================================
# 참고
# =============================================================================
1) pytohn dict in list
for p in punctuations:
    print(p['p'])
    
2) predict_porba, https://subinium.github.io/MLwithPython-2-4/
 - sklearn에서 불확실성을 추정할 수 있는 함수 (1. predict_proba, 2. decision_function)
 - 각 클래스에 대한 확률.
 - 각 행의 첫 번째 원소는 첫 번째 클래스의 예측 확률, 두 번째 원소는 두 번째 클래스의 예측 확률 (출력은 항상 0~1 사이의 값.)

3) calibration, https://3months.tistory.com/490
         LeNet, http://www.hellot.net/new_hellot/magazine/magazine_read.html?code=202&sub=200&idx=42919&ver=1
        ResNet, https://m.blog.naver.com/laonple/221259295035
        
4) train['text'][0].replace('.', ' {} '.format('.'))