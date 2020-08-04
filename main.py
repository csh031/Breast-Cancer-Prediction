import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mpld3 as mpl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df = pd.read_csv("datasets_180_408_data.csv", header = 0)#header가 없으면 인덱스를 자동으로 만듦 
df.drop('id', axis=1, inplace=True)#axis가 1이면 행에서 삭제
df.drop('Unnamed: 32', axis=1, inplace=True)#inplace가 True면 기존 데이터를 수정된 데이터로 덮어쓰기
print(df.diagnosis.unique())#diagnosis 행에 있는 값들을 중복 허용 안하고 불러옴(M,N) M이 악성종양, N은 그냥 종양
df['diagnosis'] = df['diagnosis'].replace(['M','B'],[1,0])#M과 B을 각각 1과 0으로 바꿈
print(df.head())

#데이터 시각화, radius, perimeter, area, compactness, concavity, concave points는 악성과 그냥 종양이 구분됨
#나머지는 구분에 사용하기 힘들다
features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]

plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    #bins는 몇개의 구간으로 할지, 왼쪽거는 표현하고 싶은 값, alpha는 투명도, normed는 정규화, stacked는 그래프에서 사각형들을 서로 이어서 그릴지 떨어뜨려서 그릴지 결정
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth),
     alpha=0.5,stacked=True, density=True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

data_x=df[['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean']].values
data_y=df['diagnosis'].values
print(data_x[:5])
print(data_y[:5])

(X_train, X_test, y_train, y_test) = train_test_split(data_x, data_y, train_size=0.7)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

model=Sequential()
model.add(Dense(2, input_dim=6, activation='softmax'))
sgd=optimizers.SGD(lr=0.01)
model.compile(optimizer='adam' ,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=1, epochs=200, validation_data=(X_test, y_test))
print("\n 테스트 정확도: %.4f입니다. 아님말구ㅎㅎ" % (model.evaluate(X_test, y_test)[1]))


model.save('projectmodel.h5')
