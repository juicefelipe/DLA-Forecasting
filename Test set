#Using tuned models from previously we will generate test predictions and metrics
full_X = X[:'2019-07-01']
full_y = y[:'2019-07-01']
scaler = StandardScaler()
full_X_scaled = scaler.fit_transform(full_X)
X_test_scaled = scaler.transform(X_test)
test_regression = LinearRegression(normalize = False)
test_regression.fit(full_X_scaled, full_y)
test_regression_preds = test_regression.predict(X_test_scaled)
print('Final Regression average Test Data error: $', round(np.sqrt(mean_squared_error(y_test, test_regression_preds)), 2), 'RMSE')
print('Final Regression average Test Datapercent error:', round(MAPE(y_test, test_regression_preds), 2), '%')
test_ridge = Ridge(alpha = 1, normalize = False)
test_ridge.fit(full_X_scaled, full_y)
test_ridge_preds = test_ridge.predict(X_test_scaled)
print('Final Ridge average Test Data error: $', round(np.sqrt(mean_squared_error(y_test, test_ridge_preds)), 2), 'RMSE')
print('Final Ridge average Test Data percent error:', round(MAPE(y_test, test_ridge_preds), 2), '%')
test_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
test_net.fit(full_X_scaled, full_y)
test_net_preds = test_net.predict(X_test_scaled)
print('ElasticNet average Test Data error: $', round(np.sqrt(mean_squared_error(y_test, test_net_preds)), 2), 'RMSE')
print('Final ElasticNet average Test Data percent error:', round(MAPE(y_test, test_net_preds), 2), '%')

#Let's get our final visual
fig, ax = plt.subplots(figsize = (20,10))
ax.plot(X[:'2020-03-01'].index, y[:-2], color = 'gray')
ax.plot(X_test.index, y_test, color = 'black', label = 'actual')
ax.set_xlabel('Time')
ax.set_ylabel('arc')
ax.plot(X_test.index, test_regression_preds, color = 'r', label = 'Regression')
ax.plot(X_test.index, test_ridge_preds, color = 'g', label = 'Ridge')
ax.plot(X_test.index, test_net_preds, color = 'y', label = 'Net')
plt.legend()
plt.show()

#All percent errors are acceptable, however Elastic Net still performs the best, making it a great candidate for our final model.
#Also acceptabl could be keeping all 3 final models and averaging their predictions.
