# -*- coding: utf-8 -*-
# placement_Prediction.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("placement.csv")

df.head(10)

plt.scatter(df['cgpa'],df['package'])
plt.xlabel("CGPA")
plt.ylabel("Package (in Lacs)")
plt.title("CGPA Vs Package")

X=df.iloc[:,0:1]
Y=df.iloc[:,-1]

X

Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,Y_train)

X_test

Y_test

lr.predict(X_test.iloc[0].values.reshape(1,1))

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel("CGPA")
plt.ylabel("Package (in Lacs)")
plt.title("CGPA Vs Package")

m = lr.coef_

c=  lr.intercept_

m

c

m*5.10 + c

