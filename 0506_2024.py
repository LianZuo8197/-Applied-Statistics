import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
from pandas import set_option
from numpy import arange
from pandas.plotting import scatter_matrix
names = ['x','y','z']
data1 = read_csv('reliHW10.csv')
print(data1.shape)
print(data1.dtypes)

# Set the display width
pd.set_option('display.max_colwidth', 50)
print(data1.head())
# Set the display format to two decimal places
pd.set_option('display.float_format', '{:.2f}'.format)
print(data1.describe())

#correlation
print(data1.corr(method='pearson'))

# Figures
# histgram
import matplotlib.pyplot as plt
data1.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
plt.show()

# density plot
data1.plot(kind='density',subplots=True,layout=(2,2),sharex=False,fontsize=1)
plt.show()

# box plot
data1.plot(kind='box',subplots=True,layout=(2,2),sharex=False,fontsize=2)
plt.show()

# scatter matrix plot
scatter_matrix(data1)
plt.show()

#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(data1.corr(),vmin=-1,vmax=1,interpolation='none')
# fig.colorbar(cax)
# ticks = np.arange(0,14,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
# plt.show()


# split the data
array = data1.values
X = array[:,0:1]
Y = array[:,2]
validation_size = 0.2
seed = 2024

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
                                                   test_size=validation_size,random_state=seed)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import  Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import  mean_squared_error

# evaluation baseline
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
models = { }
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()

results = [ ]
for key in models:
    kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
    cv_result = cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('{}:{} ({})'.format(key,cv_result.mean(),cv_result.std()))

fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.show()

