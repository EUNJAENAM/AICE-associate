# 1. Pandas 

### 1-1. csv파일 불러오기
```python
import pandas as pd
pd.read_csv('example.csv',sep=',')
```
### 기초
```python

df.columns    #컬럼명
df.values     #값
df.isnull().sum()     # Null 값 갯수 확인
df.describe()   #통계값(숫자값만 처리)

df.drop('customerID', axis = 1, inplace=True)   #컬럼삭제
df['TotalCharges'].astype(float)     #데이터 타입 변경

# Boolean indexing검색
cond = (df['TotalCharges'] == '') | (df['TotalCharges'] == ' ')
df[cond]

# 데이터 값 변경   ' '을 0으로
df['TotalChanges'].replace(' ', 0, inplace=True)

df['Churn'].replace(['Yes', 'No'], [1, 0], inplace=True)

# 데이터타입 일괄 변경 예제

sel_col = df.select_dtypes('object').columns

df[sel_col] = df[sel_col].astype(float)
df2.info()

# Number type의 데이터만 보기
df.select_dtypes('number').head(3)

# 컬럼의 분포 확인
df['Churn'].value_counts()
```
### Seaborn
```python
import seaborn as sns

sns.histplot(data=df, x='tenure')
sns.histplot(data=df, x='tenure', hue='Churn')
sns.kdeplot(data=df, x='tenure', hue='Churn')
sns.histplot(data=df, x='TotalCharges')
sns.kdeplot(data=df, x='TotalCharges', hue='Churn')
sns.countplot(data=df, x='MultipleLines',  hue='Churn')
df[['tenure','MonthlyCharges','TotalCharges']].corr()

sns.heatmap(df[['tenure','MonthlyCharges','TotalCharges']].corr(), annot=True)
sns.boxplot(data=df, x='Churn', y='TotalCharges')

# 구간별 나눈후 bar chart
df['Age_cat'] = pd.cut(df['Age'],
                       # 0~3, 3~7, 7~15, 15~30, 30~60, 60~80 
                       bins=[0,3,7,15,30,60,80],
                       include_lowest=True,
                       labels=['baby','children','teenage', 'young','adult','old'])
plt.figure(figsize=[14,4])
plt.subplot(1,3,1)
sns.barplot(data=df, x='Pclass', y='Survived')
plt.subplot(132)
sns.barplot(data=df, x='Age_cat', y='Survived')
plt.subplot(133)
sns.barplot(data=df, x='Sex', y='Survived')
# plt.subplots_adjust(top=1, bottom=0.1, left=0.10, right=1, hspace=0.5, wspace=0.5)
plt.show()

#--
f, ax = plt.subplots(1,2, figsize=(12,6)) 
sns.countplot('Sex', data=df, ax=ax[0])
sns.countplot('Sex', hue='Survived', data=df, ax=ax[1])
plt.show()
```

### 1-2. 결측치 채우기
```python
df.fillna('채울값', inplace=True)
df['col'] = df['col'].fillna(df['col'].mode()[0])   #최빈값
```

### 1-3. 결측치가 있는 행 삭제하기
```python
df.dropna(axis=0, inplace=True)
```

### 1-4. 특정 행 삭제하기
```python
df.drop(삭제할 행 index, axis=0)
```

### 1-5. 특정 컬럼 삭제하기
```python
df.drop(삭제할 컬럼명, axis=1)
# 삭제할 컬럼이 여러 개일 경우 리스트 사용
df.drop([컬럼명1,컬럼명2] axis=1)  
```
### 1-6 더미 변수 타입으로 변환(One-hot encoding)
```python
pd.get_dummies(data=df, columns=['컬럼명'])

# object타입 일괄 더미변수타입으로 변환
cal_cols = df.select_dtypes('object').columns.values
pd.get_dummies(data=df, columns=cal_cols)

```

### chart
```python
import matplotlib.pyplot as plt
%matplotlib inline

df['gender'].value_counts().plot(kind='bar')

```


# 2. Scikit-learn

### 2-1. 데이터셋 분할하기

```python
#특정 컬럼제외하고 X값 저장
X = df.drop('Churn', axis=1).values
y = df1['Churn'].values

from sklearn.model_selection import train_test_split

# Feature데이터와 label데이터가 분리되어 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# Feature데이터와 label데이터가 하나의 데이터프레임에 같이 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)

# Feature데이터와 label데이터가 하나의 데이터프레임에 같이 있는 경우, (레이블까지 고려하여) 데이터셋 균등하게 나누기
train_x, test_x, train_y, test_y = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# 데이터셋 분할 후, 분할된 데이터셋 확인
train_x.shape, test_x.shape, train_y.shape, test_y.shape
```

### 2-2. 데이터 정규화 - MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
```

### 2-3. 데이터 정규화 - StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
```

### 2-4. 머신러닝 모델 - 분류 : 랜덤포레스트 분류기
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=3, random_state=123)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```
### 2-4-1 머신러닝 모델 - 성능 측정(accuracy, precision, recall, f1)
```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

print(accuracy_score(test_y, pred_y))
print(precision_score(test_y, pred_y))
print(recall_score(test_y, pred_y))
print(f1_score(test_y, pred_y))
```
### 2-4-2 머신러닝 모델 - confusion_matrix를 heatmap으로 출력하기
```python
from sklearn.metrics import confusion_matrix
data = confusion_matrix(test_y, pred_y)

# seaborn으로  그려보자.
sns.heatmap(data, annot=True, cmap='Reds')
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()

# 상관관계 heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between features')
plt.show()
```
### 2-4-3 머신러닝 모델 - classification_report로 성능 확인하기
```python
from sklearn.metrics import classification_report

classification_report(test_y, pred_y, target_names=target_names)
```

### 2-5. 머신러닝 모델 - 분류 : 로지스틱 회귀
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model = LogisticRegression()     #객체 생성
model.fit(train_x, train_y)      #훈련
model.score(test_x, test_y)      #성능평가
pred_y = model.predict(test_x)   #예측

confusion_matrix(y_test, prid_y)  #오차행렬
accuracy_score(y_test, prid_y)    #정확도
precision_score(y_test, prid_y)   #정밀도
recall_score(y_test, prid_y)      #재현율
f1_score(y_test, prid_y)          #정밀도+재현율
print(classification_report(y_test, lg_pred))
   
```

### 2-6. 머신러닝 모델 - 분류 : K-Nearest Neighbor
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-7. 머신러닝 모델 - 분류 : DecisionTree
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(randon_state=42)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-8. 머신러닝 모델 - 분류 : SGDClassifier
```python
model = SGDClassifier(random_state=123, learning_rate='optimal')
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-9. 머신러닝 모델 - 분류 : xgboost
```bash
pip install xgboost
```

```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=3, random_state=42) 
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-10. 머신러닝 모델 - 분류 : LGBM
```bash
pip install lightgbm
```

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=3, random_state=42) 
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-11. 머신러닝 모델 - 회귀 : LinearRegression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-12. 머신러닝 모델 - 회귀 : RandomForestRegressor
```python
from sklearn.linear_model import RandomForestRegressor

# train_y가 Pandas 데이터프레임이거나, Pandas Series인 경우
# 다차원 배열을 1차원 배열로 만들기
train_y = train_y.to_numpy().flatten()

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-13. 머신러닝 모델 - 회귀 : GradientBoostingRegressor
```python

# train_y가 Pandas 데이터프레임이거나, Pandas Series인 경우
# 다차원 배열을 1차원 배열로 만들기
train_y = train_y.to_numpy().flatten()

model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,random_state=42)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-14. 머신러닝 모델 - 회귀 : XGBRegressor
```python
from xgboost import XGBRegressor as xgb

# train_y가 Pandas 데이터프레임이거나, Pandas Series인 경우
# 다차원 배열을 1차원 배열로 만들기
train_y = train_y.to_numpy().flatten()

model = XGBRegressor(n_estimators=100,gamma=1,reg_lambda=5, reg_alpha=5,random_state=42)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

### 2-15. 머신러닝 모델 - 회귀 : Voting Ensemble Model
```python
from sklearn.linear_model import LinearRegression as lr
from xgboost import XGBRegressor as xgb
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingRegressor as grb
from sklearn.ensemble import VotingRegressor

import joblib
import time

model_list=[lr(), rfr(), grb(), xgb()]

# train_y가 Pandas 데이터프레임이거나, Pandas Series인 경우
# 다차원 배열을 1차원 배열로 만들기
train_y = train_y.to_numpy().flatten()

# 개별 모델들을 학습한 후에 모델을 저장한다.
model_result = []
for i in range(len(model_list)):
    model = model_list[i]
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    model_result.append(model)

# 개별 모델들을 Tuple타입으로 리스트에 넣고
voting_models = [
    ('linear_reg', model_rslt[0]), 
    ('randForest', model_rslt[1]), 
    ('gradBoost', model_rslt[2]), 
    ('xgboost', model_rslt[3])
]

# VotingRegressor를 하나의 모델을 학습 하듯이 학습하고 추론한다.
voting_regressor = VotingRegressor(voting_models, n_jobs=-1)
voting_regressor.fit(train_x, train_y)
pred_y = voting_regressor.predict(test_x)    
```

# 3. Matplotlib & Seaborn


# 4. Tensorflow(Keras)

### 4-1. 딥러닝 모델 - 심층신경망 만들기(회귀 모델)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.random import set_seed

set_seed(100)

col_num = train_x.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(col_num,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()      #모델 요약

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])    
              
```

### 4-2. 딥러닝 모델 - 심층신경망 만들기(이진 분류 모델)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(100)

col_num = train_x.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(col_num,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',   
              optimizer='adam',
              metrics=['accuracy'])      
model.summary()

# 학습
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=10)

```
### dropout 설계
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(39,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

```

### 4-3. 딥러닝 모델 - 심층신경망 만들기(다중 분류 모델)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(100)
class_num = 2

# 회귀 모델의 경우
col_num = train_x.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(col_num,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))

# Y값을 One-Hot-Encoding 하지 않은경우
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  
              
# Y값을 One-Hot-Encoding 한 경우
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  
              
model.summary() 

# 모델 학습
history = model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
          epochs=20, 
          batch_size=16)
```

### 4-4. 딥러닝 모델 - 콜백 함수 설정 (조기종료)
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# val_loss 모니터링해서 성능이 5epoch동안 val_loss가 낮아지지 않으면 조기 종료
early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                           verbose=1, patience=5)

# val_loss 가장 낮은 값을 가질때마다 모델저장
check_point = ModelCheckpoint('best_model.h5', verbose=1, 
                              monitor='val_loss', mode='min', save_best_only=True)

# fit()함수의 callbacks에 전달할 리스트
callbacks = [early_stop, check_point]
```

### 4-5. 딥러닝 모델 - 학습
```python
history = model.fit(x=X_train, y=y_train, epochs=50 , batch_size=20,
          validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)
```

### 4-6. 딥러닝 모델 - 성능 평가 - loss 그래프 그리기

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'])
plt.show()
```

### 성능평가
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
pred = model.predict(X_test)
y_pred = np.argmax(pred, axis=1)
#score
객체.score(X_test, y_text)
# 정확도 80%
accuracy_score(y_test, y_pred)
# 정밀도
precision_acore(y_test, y_pred)
# 재현율 성능이 좋지 않다
recall_score(y_test, y_pred)
# 정밀도+재현율
f1.acore(y_test, y_pred)
# accuracy, recall, precision 성능 한번에 보기
print(classification_report(y_test, y_pred))

```
### 다중분류시 예측값
```python
y_pred = np.argmax(pred, axis=1)
accuracy_score(y_test, y_pred)

```
