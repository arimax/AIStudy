import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#wineの学習データを取得する
df = pd.read_csv('./wine_train.csv',header=0)
# 最後のqualityだけ抜いておく
X = df.iloc[:,:-1]
y = df.iloc[:,[-1]]
print(y.shape)
print(X.shape)

from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
# Classifierなので、GradientBoostingClassifierをselectorとして使用する
# 特徴量を10,1回のstepで削除する次元は5%ずつ
selector = RFE(estimator=GradientBoostingClassifier(random_state=0),
               n_features_to_select=10,
               step=0.05)
selector.fit(X,y.values.ravel())

from sklearn.preprocessing import Imputer
#欠損値NaNを平均で置き換える axisは平均値を横で取るのか縦で取るのか（縦で取る）
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
# fitで学習する
imp.fit(X)
X_columns = X.columns.values
#transformで変換 NumpyArrayで返ってくる
X = pd.DataFrame(imp.transform(X),columns=X_columns)
X.head()

# 残っているかどうかをprint
print(selector.support_)

# transformすると、絞り込んだ結果を出してくれる
X_selected = selector.transform(X)
X_selected = pd.DataFrame(X_selected,columns=X_columns[selector.support_])
print(X_selected.shape)
print(X_selected.dtypes)

#いくつかの方法で検証
pipeline_dict = {}
pipeline_dict['KNeighborsClassifier'] = Pipeline([('scl',StandardScaler()),('est',KNeighborsClassifier())])
pipeline_dict['GradientBoostingClassifier'] = Pipeline([('scl',StandardScaler()),('est',GradientBoostingClassifier(random_state = 42))])
pipeline_dict['LogisticRegression'] = Pipeline([('scl',StandardScaler()),('est',LogisticRegression(random_state=1))])
pipeline_dict['RandomForestClassifier'] = Pipeline([('scl',StandardScaler()),('est',RandomForestClassifier(random_state=1))])

#wineのテストデータを取得する
df = pd.read_csv('./wine_test.csv',header=0)
X_test = df.iloc[:]
print(X.shape)

X.describe()

# fitで学習する(imputer)
imp.fit(X_test)
X_test_columns = X_test.columns.values
#transformで変換 NumpyArrayで返ってくる
X_test = pd.DataFrame(imp.transform(X_test),columns=X_test_columns)
X_test.head()
X_test_selected = selector.transform(X_test)
X_test_selected = pd.DataFrame(X_test_selected,columns=X_test_columns[selector.support_])
print(X_test_selected.shape)
print(X_test_selected.dtypes)

import csv
from sklearn.metrics import roc_auc_score

#hold_outを行って、accuracyを検証する
X_train,X_vali,y_train,y_vali = train_test_split(X,y,test_size=0.2,random_state=1)

#一番有効な方法が何か調べる
for name,pipeline in pipeline_dict.items():
    pipeline.fit(X_train,y_train.values.ravel())
    y_predict = pipeline.predict(X_vali)
    
    print('estimation:%s, score:%0.6f' % (name,accuracy_score(y_vali,y_predict)))
    print('estimation:%s, score:%0.6f' % (name,roc_auc_score(y_vali,y_predict)))

for name,pipeline in pipeline_dict.items():
    pipeline.fit(X,y.values.ravel())
    y_predicts = pipeline.predict(X_test)
    f = open(name +'.csv', 'w')
    writer = csv.writer(f, lineterminator='\n') 
    #300行・1列に変換
    y_t = y_predicts.reshape(y_predicts.size,1)
    writer.writerows(y_t)
    f.close