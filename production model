#Now for the final production model
new_X = new_data.drop('arc', axis = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
new_data_scaled = scaler.transform(new_X)
final_regression = LinearRegression(normalize = False)
final_regression.fit(X_scaled, y)
final_regression_preds = final_regression.predict(new_data_scaled)
final_ridge = Ridge(alpha = 1, normalize = False)
final_ridge.fit(X_scaled, y)
final_ridge_preds = final_ridge.predict(new_data_scaled)
final_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
final_net.fit(X_scaled, y)
final_net_preds = final_net.predict(new_data_scaled)

#Plot the final predictions for the rest of 2020:
fig, ax = plt.subplots(figsize = (20,10))
ax.plot(X.index, y, color = 'black', label = 'actual')
ax.set_xlabel('Time')
ax.set_ylabel('arc')
ax.plot(new_data.index, final_regression_preds, color = 'r', label = 'Regression')
ax.plot(new_data.index, final_ridge_preds, color = 'g', label = 'Ridge')
ax.plot(new_data.index, final_net_preds, color = 'y', label = 'Net')
plt.legend()
plt.show()

#We will also add prediction columns to the dataset for use in Domo and export.
new_data['reg_preds'] = final_regression_preds.astype('int64')
new_data['ridge_preds'] = final_ridge_preds.astype('int64')
new_data['net_preds'] = final_net_preds.astype('int64')
new_data.reset_index(inplace = True)

domo.write_dataframe(new_data, 'output preds')
