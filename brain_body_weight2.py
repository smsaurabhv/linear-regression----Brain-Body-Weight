import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

# read data

dataframe  =pd.read_csv('newdata.csv',header=None)
x_values = dataframe.iloc[:,:-1].values
y_values =dataframe.iloc[:,1].values


print("x values is")
print(x_values)

print("y values is")
print(y_values)

bodyreg  = linear_model.LinearRegression()
bodyreg.fit(x_values,y_values)

ypred=bodyreg.predict(x_values)
print("total prediction")
print(ypred)

print("coefficient")
print(bodyreg.coef_)

print("intercept")
print(bodyreg.intercept_)


print("total error")
print(np.sum(np.absolute(ypred-y_values)**2))

plt.scatter(x_values,y_values)
plt.plot(x_values,bodyreg.predict(x_values))
plt.show()

