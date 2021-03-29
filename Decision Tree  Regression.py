#!/usr/bin/env python
# coding: utf-8
# Decision tree - X (independent_var) and y (dependent_var)


# Importing the libraries
import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd  


# Importing the dataset
df = pd.read_csv(r"your directory")
df.describe()


# Assigning the input and output values resp. dividing data into attributes and labels
y = df.iloc[0:, 1].values  # (dep_v)
X = df.iloc[0:, 35].values # (indep_v)


y = np.array(y).astype('float')
X = np.array(X).astype('float')
X = X.reshape(-1,1)

# Check the values
X
y


# Divide the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# Import the regressor 
from sklearn.tree import DecisionTreeRegressor  
  
# Create the regressor object 
regressor = DecisionTreeRegressor(random_state=101)  
  
# Fit the regressor with X and y data and make prediction
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test)

# Check predicted values
y_pred
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# Bar chart - actual and predicted values
df1 = df.head(68)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.2', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
plt.show()


# Correlation with regression
sns.pairplot(df, kind="reg")
# Density
sns.pairplot(df, diag_kind="kde")
plt.show()


# Evaluating the algorithm using the scikit library
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
import sklearn.metrics as metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_test - y_pred)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Calculate and print MAPE again
def MAPE(y_pred, y_test):
    return ( abs((y_test - y_pred) / y_test).mean()) * 100
print ('My MAPE: ' + str(MAPE(y_pred, y_test)) + ' %' )


# Plotting
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


#Plotting
X_grid = np.arange(min(X), max(X), 0.01)   
X_grid = X_grid.reshape((len(X_grid), 1))  
  
# Scatter plot for original data 
plt.scatter(X, y, color = 'red') 
  
# Plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')  
  
# Specify title 
plt.title('Decision Tree')  
  
# Specify X axis label 
plt.xlabel('X-Feature') 
  
# Specify y axis label 
plt.ylabel('y-Label') 
  
# Show the plot 
plt.show() 


# Plotting of prediction
plt.plot(y_pred, label='Prediction')
plt.show()


# Plotting of predicted and actual values for better understanding
plt.plot(y_test, label='Actual values')
plt.plot(y_pred, color="red", label='Predicted values')
plt.legend()
plt.show()


# Jointplot - to visualize how data is distributed
import seaborn as sns
sns.jointplot(x=y_test, y=y_pred, kind='kde')
#plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.show()


# Lineplot
sns.lineplot(x=y_test, y=y_pred)
plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.show()


# Boxplot - actual and predicted values, how does the algorithm manage outliers
import seaborn as sns
sns.boxplot(x=y_test.flatten(), y=y_pred.flatten()) 
plt.ylabel('Predicted', fontsize=14)
plt.xlabel('Actual', fontsize=14)
plt.show()


# Decision tree using scikit-learn
from sklearn import tree
tree.plot_tree(regressor);

