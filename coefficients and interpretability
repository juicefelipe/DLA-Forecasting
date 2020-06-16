#Coefficients help us gain insight into how the model is impacted by each column. When a predictor variable increases by 1 
#this coefficient is how much we anticipate the ARC (target variable) to go up/down, all else held constant.

##For Example: In the Regression Model, for each $1 increase in PY production we anticipate 0.65 dollar increase in ARC, all else 
##held constant. Whereas we anticipate a $422512.58 decrease in ARC for every 1% increase in unemployment rate, all else held constant.

#Thus, the end user who may already have some insight to a significant increase in unemployment rate, or number of upcoming hires/layoffs, 
#or high activity_count standards for the month can can calculate what the expected impact on ARC will be according to the model.

column_means = X.mean()
column_stds = X.std()

regression_coefs = (final_regression.coef_.tolist()) / column_stds
ridge_coefs =  final_ridge.coef_.tolist() / column_stds
net_coefs = final_net.coef_.tolist() / column_stds

coefficient_df = pd.DataFrame({'Regression Coefficient': round(regression_coefs, 2), 
                               'Ridge Coefficient': round(ridge_coefs, 2), 'Elastic Net Coefficient': round(net_coefs, 2)})
coefficient_df

#Note that division by column_stds was necessary for correct interpretation since the data had been scaled before fitting.
