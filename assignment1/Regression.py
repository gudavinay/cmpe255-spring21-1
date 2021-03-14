import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('Data/housing.csv', header=None, delim_whitespace=True, names=features)

# print(df.head())

# Missing value identification
# print(df.isna().sum())     
# print(df.isnull().sum())
# None identified

regression = LinearRegression()
RMSE, RSquared = [], []
for i in range(len(features)-1):
    X = df.iloc[:, i:i+1].values
    Y = df.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = None)
    regression.fit(X_train, Y_train)
    Y_pred = regression.predict(X_test)
    RMSE.append(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    RSquared.append(metrics.r2_score(Y_test, Y_pred))

# print(RMSE)
# print(RSquared)

# ['CRIM',              'ZN',                'INDUS',             'CHAS',               'NOX',               'RM',               'AGE',                'DIS',              'RAD',               'TAX',               'PTRATIO',           'B',                 'LSTAT',          
# [8.35454696472828,    9.356026269748346,   7.063289050277013,   8.226526203695428,    6.961628201563447,   7.634387800178865,  8.43520716033072,     9.294077823762919,  8.491215905933377,   8.947110143089377,   7.380087745063264,   8.525005484986657,   6.742436554542911]
# [0.14113847771076737, 0.08132579873093615, 0.32617196482586597, 0.033022118773414766, 0.22635475522645698, 0.4257244536389171, 0.060768680449074575, 0.0917311655985581, 0.09154540519931809, 0.15007389100143476, 0.20889493213653076, 0.10332756915383035, 0.5540704641419649]

# print(RMSE[12])
# print(RSquared[12])

#LSTAT and RM look promising - proceeding with LSTAT

def Linear_Regression():
    X = df.iloc[:, 12:13].values
    y = df.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    regression = LinearRegression()
    regression.fit(X_train, Y_train)

    Y_pred = regression.predict(X_test)
    print("Linear Regression\nRMSE",np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print("RSquared",metrics.r2_score(Y_test, Y_pred))
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test, Y_test, marker='o')
    plt.plot(X_test, Y_pred,color='red')
    plt.title('LSTAT vs House Price')
    plt.xlabel('LSTAT')
    plt.ylabel('House Price')
    plt.savefig('LinearRegression.png')
    plt.show()

def Polynomial_Regression(deg):
    X = df.iloc[:, 12:13].values
    y = df.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    polynomial_regression = PolynomialFeatures(degree = deg)
    X_poly = polynomial_regression.fit_transform(X_train)
    # print(X_poly)
    regression = LinearRegression()
    regression.fit(X_poly, Y_train)
    Y_pred = regression.predict(polynomial_regression.fit_transform(X_test))

    print("\nPolynomial Regression with degree",deg,"\nRMSE",np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print("RSquared",metrics.r2_score(Y_test, Y_pred))

    plt.figure(figsize=(10, 5))
    plt.scatter(X_test, Y_test, marker='o')
    plt.plot(X_test, Y_pred,color='red')
    plt.title('LSTAT vs House Price')
    plt.xlabel('LSTAT')
    plt.ylabel('House Price')
    plt.savefig('PolynomialRegressionDegree'+str(deg)+'.png')
    plt.show()

def Multiple_Regression():

    X = df[['LSTAT','RM','TAX']]
    # print(X.head())
    y = df.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    regression = LinearRegression()
    regression.fit(X_train, Y_train)

    Y_pred = regression.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    RSquared = metrics.r2_score(Y_test, Y_pred)
    AdjustedRSquared = (1 - (1-RSquared)*(len(Y_test)-1)/(len(Y_test)-X_train.shape[1]-1))
    print("\nMultiple Regression\nRMSE",RMSE,"\nRSquared",RSquared,"\nAdjustedRSquared",AdjustedRSquared)

if __name__ == "__main__":
    Linear_Regression()
    Polynomial_Regression(2)
    Polynomial_Regression(20)
    Multiple_Regression()