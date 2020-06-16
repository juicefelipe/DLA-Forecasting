#Import data from Domo and view the information summary

import domojupyter as domo
df = domo.read_dataframe('Rev & Exp test', query='SELECT * FROM table')
print(df.shape)
print(df.info())

#Notice the 'object' Dtypes, we need to correct them to float64 for TOM1, and to datetime for reportmonth
import pandas as pd
df['TOM1'] = df['TOM1'].str.replace("'", "")
df['TOM1'] = df['TOM1'].astype('float64')
df['reportmonth'] = pd.to_datetime(df['reportmonth'])
df.info()

#Filter for just DLA
dla = df[df['agencyname']=='Dixie Leavitt Agency']

#Do we have any missing data? Display the proportions of Null Values.
print(dla.isnull().mean().sort_values(ascending = False))

#Where are all the nulls coming from?
dla.tail(8)

#Since the only nulls are the most recent months. For now we will assume they just have not yet been recorded.

##This is intended to be a forecast of the future, thus only columns that can be projected into the future, 
##plus the target column 'arc', are kept:
##Set Report Month as the new index and keep PY Production & PY Total Business, Avg US Unemployment Rate, 
##Activity Count, Days Since, Employee Count, and ARC
dla.set_index('reportmonth', inplace = True)
dla_select = dla[['pyproduction', 'pytotalbusiness', 'avg_unemployment_rate_us', 'activity_count', 'days_since', 'Employee_Count', 'arc']]
print(dla_select.shape)
print(dla_select.info())

#Also view the correlations matrix. We don't want to include redundant information if we don't need to.
dla_select.corr()

#Notice the nearly perfect correlation between pyproduction and pytotalbusiness. 
#We will use only one of these because the information is nearly identical. For now, we will drop pytotalbusiness.
dla_select_drop = dla_select.drop('pytotalbusiness', axis = 1)

#Manual updates to the unemployment rate column are added to hold all of what we know about covid-19 so far, 
#plus a more realistic estimate of the future.
dla_select_drop.loc['2020-03-31', 'avg_unemployment_rate_us'] = 4.4
dla_select_drop.loc['2020-04-30', 'avg_unemployment_rate_us'] = 14.7
dla_select_drop.loc['2020-05-31', 'avg_unemployment_rate_us'] = 13.3
dla_select_drop.loc['2020-06-30', 'avg_unemployment_rate_us'] = 13
dla_select_drop.loc['2020-07-31', 'avg_unemployment_rate_us'] = 13
dla_select_drop.loc['2020-08-31', 'avg_unemployment_rate_us'] = 13
dla_select_drop.loc['2020-09-30', 'avg_unemployment_rate_us'] = 13
dla_select_drop.loc['2020-10-31', 'avg_unemployment_rate_us'] = 13
dla_select_drop.loc['2020-11-30', 'avg_unemployment_rate_us'] = 13
dla_select_drop.loc['2020-12-31', 'avg_unemployment_rate_us'] = 13

#Consider the scale of each column. Large differences in scale could mean one column dominates the model weights. 
#Looking here at the columns variance we see there are huge scale differences
dla_select_drop.var()

#We will scale each column appropriately to have mean 0 and variance 1 after separating into training and testing data sets,
#but first split into historical and future data.
historical = dla_select_drop[:'2020-05-01']
new_data = dla_select_drop['2020-05-01':]
new_data.head()

#There are 2 columns in the newdata set that still have nulls that we need to fix, activity_count and pyproduction. 
#Activity_Count we will fill with the historical average and PYproduction will be filled with last year's CYproduction.
dla_select_drop['activity_count'].fillna((dla_select_drop['activity_count'].mean()), inplace = True)
vals = dla['2019-05-01':'2019-12-31'].cyproduction.tolist()
new_data.pyproduction = vals

#Split the historical data into training, validation, and testing sets for use in models.
X = historical.drop(['arc'], axis = 1)
y = historical['arc']
X_train = X[:'2019-01-01']
X_val = X['2019-01-01':'2019-07-01']
X_test = X['2019-07-01':'2020-02-29']
y_train = y[:'2019-01-01']
y_val = y['2019-01-01':'2019-07-01']
y_test = y['2019-07-01':'2020-02-29']
