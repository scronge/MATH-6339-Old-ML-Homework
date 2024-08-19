
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Define the training data
data = np.array([[1, 3],
                [-1, -2],
                [2, 4]])

# Print the matrix 'tab'
print(data)


rownames = ["1", "2", "3"]
colnames = ["x", "t"]

Data1 = pd.DataFrame(data, index=rownames, columns=colnames)

print(Data1)

# Create a linear regression model
X = Data1['x']
y = Data1['t']
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())



# Assuming you have already fit the model as shown in your code
# Predict the values based on the linear regression model
predicted_values = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, predicted_values)

# Extract the data for X and y
X = X.squeeze()
y = y.squeeze()

# Fit a simple linear regression model to calculate slope and intercept
X_with_intercept = sm.add_constant(X)
model = sm.OLS(y, X_with_intercept).fit()
slope, intercept = model.params

# Create a scatter plot of the original data points
plt.scatter(X, y, label='Data Points')

# Plot the regression line
plt.plot(X, predicted_values, color='red', label=f'Linear Regression Line (R^2 = {r2:.2f})')

# Create the equation of the line
line_eq = f'Y = {slope:.2f}X + {intercept:.2f}'

# Add labels and a legend
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Display the line equation below the label
plt.text(0.2, 0.65, line_eq, transform=plt.gca().transAxes, fontsize=12, color='blue')

# Show the plot
plt.show()







