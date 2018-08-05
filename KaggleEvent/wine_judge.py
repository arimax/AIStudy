import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#wineの学習データを取得する
df = pd.read_csv('./wine_train.csv',header=0)

# 最後のqualityだけ抜いておく
X = df.iloc[:,:-1]
y = df.iloc[:,[-1]]


from sklearn.preprocessing import Imputer
#欠損値NaNを平均で置き換える axisは平均値を横で取るのか縦で取るのか（縦で取る）
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
# fitで学習する
imp.fit(X)
X_columns = X.columns.values
#transformで変換 NumpyArrayで返ってくる
X = pd.DataFrame(imp.transform(X),columns=X_columns)
X.head()

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

import csv
from sklearn.metrics import roc_auc_score

#hold_outを行って、accuracyを検証する
X_train,X_vali,y_train,y_vali = train_test_split(X,y,test_size=0.2,random_state=1)

#パラメータサーチを行う。
import xgboost as xgb
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV

# モデルにインスタンス生成
xgb = xgb.XGBClassifier(nthreads=-1)

#パラメーターをRandomSearchする
params = {  
            "n_estimators": st.randint(3, 40),
            "max_depth": st.randint(3, 40),
            "learning_rate": st.uniform(0.05, 0.4),
            "colsample_bytree": st.beta(10, 1),
            "subsample": st.beta(10, 1),
        }

random_search = RandomizedSearchCV(xgb,params,scoring='roc_auc')
random_search.fit(X_train, y_train.values.reshape(-1,))
y_predict = random_search.predict(X_vali)
print('score:%0.6f' % (accuracy_score(y_vali,y_predict)))
print('score:%0.6f' % (roc_auc_score(y_vali,y_predict)))

#こちらが本番
y_predict = random_search.predict(X)
f = open('result.csv', 'w')
writer = csv.writer(f, lineterminator='\n') 
#300行・1列に変換
y_t = y_predict.reshape(y_predict.size,1)
writer.writerows(y_t)
f.close
