#Using the prepared data in the train/validation/test sets we will scale each feature to mean 0 var 1 and test multiple models.
##Linear Regression:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
lin_reg = LinearRegression()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
lin_reg.fit(X_train_scaled, y_train)
X_val_scaled = scaler.transform(X_val)
lin_preds = lin_reg.predict(X_val_scaled)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, lin_preds)), 2), 'RMSE')

def MAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('average percent error:', round(MAPE(y_val, lin_preds), 2), '%')

##Random Forest:
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train_scaled, y_train)
rf_preds = rf_reg.predict(X_val_scaled)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, rf_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, rf_preds), 2), '%')

##Support Vector Machine:
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train_scaled, y_train)
svr_preds = svr.predict(X_val_scaled)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, svr_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, svr_preds), 2), '%')

##Bayesian Model:
from sklearn.linear_model import BayesianRidge
br_reg = BayesianRidge()
br_reg.fit(X_train_scaled, y_train)
br_preds = br_reg.predict(X_val_scaled)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, br_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, br_preds), 2), '%')

##Ridge Regression:
from sklearn.linear_model import Ridge
rid_reg = Ridge()
rid_reg.fit(X_train_scaled, y_train)
rid_preds = rid_reg.predict(X_val_scaled)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, rid_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, rid_preds), 2), '%')

##Elastic Net:
from sklearn.linear_model import ElasticNet
en_reg = ElasticNet()
en_reg.fit(X_train_scaled, y_train)
en_preds = en_reg.predict(X_val_scaled)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, en_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, en_preds), 2), '%')

#Plot the predictions:
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (20,10))
ax.plot(X[:'2019-07-01'].index, y[:'2019-07-01'], color = 'gray')
ax.plot(X_val.index, y_val, color = 'black', label = 'actual')
ax.set_xlabel('Time')
ax.set_ylabel('arc')
ax.plot(X_val.index, rf_preds, color = 'r', label = 'Random Forest')
ax.plot(X_val.index, lin_preds, color = 'g', label = 'Regression')
ax.plot(X_val.index, rid_preds, color = 'y', label = 'Ridge')
ax.plot(X_val.index, en_preds, color = 'b', label = 'ElasticNet')
plt.legend()
plt.show()

#Linear Regression, Ridge Regression, and ElasticNet may all be reasonable models. 
#Get more specific by tuning their parameters.
