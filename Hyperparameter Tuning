#Regression Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
lin_params = {'normalize':[False, True]}
lin_reg = LinearRegression()
lin_reg_cv = GridSearchCV(lin_reg, param_grid = lin_params, scoring = 'neg_root_mean_squared_error')
lin_reg_cv.fit(X_train_scaled, y_train)
lin_reg_cv_preds = lin_reg_cv.predict(X_val_scaled)
print('Using these parameters', lin_reg_cv.best_params_)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, lin_reg_cv_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, lin_reg_cv_preds), 2), '%')

#Ridge Hyperparameter Tuning
rid_params = {'normalize':[False, True], 'alpha':[0,0.3, 0.5, 0.7, 1]}
rid_reg = Ridge()
rid_reg_cv = GridSearchCV(rid_reg, param_grid = rid_params, scoring = 'neg_root_mean_squared_error')
rid_reg_cv.fit(X_train_scaled, y_train)
rid_reg_cv_preds = rid_reg_cv.predict(X_val_scaled)
print('Using these parameters', rid_reg_cv.best_params_)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, rid_reg_cv_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, rid_reg_cv_preds), 2), '%')

#Elastic Net Hyperparameter Tuning
net_params = {'alpha':[0, 0.3, 0.5, 0.7, 1], 'l1_ratio':[0, 0.3, 0.5, 0.7, 1]}
net_reg = ElasticNet()
net_reg_cv = GridSearchCV(net_reg, param_grid = net_params, scoring = 'neg_root_mean_squared_error')
net_reg_cv.fit(X_train_scaled, y_train)
net_reg_cv_preds = net_reg_cv.predict(X_val_scaled)
print('Using these parameters', net_reg_cv.best_params_)
print('average error: $', round(np.sqrt(mean_squared_error(y_val, net_reg_cv_preds)), 2), 'RMSE')
print('average percent error:', round(MAPE(y_val, net_reg_cv_preds), 2), '%')

#For comparison ease display side by side
linear = LinearRegression(normalize = False)
linear.fit(X_train_scaled, y_train)
linear_preds = linear.predict(X_val_scaled)
print('Linear Regression average error: $', round(np.sqrt(mean_squared_error(y_val, linear_preds)), 2), 'RMSE')
print('Linear Regression average percent error:', round(MAPE(y_val, linear_preds), 2), '%')
ridge = Ridge(alpha = 1, normalize = False)
ridge.fit(X_train_scaled, y_train)
ridge_preds = ridge.predict(X_val_scaled)
print('Ridge average error: $', round(np.sqrt(mean_squared_error(y_val, ridge_preds)), 2), 'RMSE')
print('Ridge average percent error:', round(MAPE(y_val, ridge_preds), 2), '%')
net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
net.fit(X_train_scaled, y_train)
net_preds = net.predict(X_val_scaled)
print('ElasticNet average error: $', round(np.sqrt(mean_squared_error(y_val, net_preds)), 2), 'RMSE')
print('ElasticNet average percent error:', round(MAPE(y_val, net_preds), 2), '%')

#It appears a tuned ElasticNet has the lowest error, but does it still perform best branching out to the test set? 
