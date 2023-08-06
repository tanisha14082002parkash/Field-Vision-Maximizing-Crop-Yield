# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 00:07:42 2023

@author: Tanisha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
'''
def func(x, a, b, c, d, e):
    return a * x*5 + b * x4 + c * x3 + d * x*2 + e * x

# Sample data points of the graph
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 1, 6, 3])

# Perform polynomial regression (curve fitting) using numpy.polyfit
degree = 2  # Set the degree of the polynomial (you can adjust this as needed)
coefficients = np.polyfit(x_data, y_data, degree)

# Create the polynomial function based on the coefficients
def fitted_function(x):
    return np.polyval(coefficients, x)

# Generate new x values for the function
x_fit = np.linspace(min(x_data), max(x_data), num=100)

# Evaluate the function at the new points
y_fit = fitted_function(x_fit)

# Plot the original data points and the fitted function
plt.scatter(x_data, y_data, label='Data Points', color='red')
plt.plot(x_fit, y_fit, label='Fitted Function', color='blue')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

df1 = pd.read_csv('preprocessed_n.csv', index_col = 'created_at', parse_dates = True)
df2= pd.read_csv('preprocessed_p (1).csv', index_col = 'created_at', parse_dates = True)

df1.plot()
df=df1['2022-11-01 00:00:00+05:30':'2022-12-12 23:00:00+05:30']
df.plot()
train= df[:'2022-11-28 00:00:00+05:30']
test=df['2022-11-28 00:00:00+05:30':]
train=df2.iloc[185000:210000,0]
test=df2.iloc[200000:220000,0]
# Sample data points of the graph
x_data = np.arange(1, 25001)
#x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array(train)

# Define the custom function you want to fit (quadratic in this case)
def quadratic_function(x, a, b, c,d):
    return a* x***3 + b * x ** 2 + c * x + d

# Perform curve fitting using scipy's curve_fit
params, _ = curve_fit(quadratic_function, x_data, y_data)

# Create the fitted function using the obtained parameters
def fitted_function(x):
    return quadratic_function(x, *params)

# Generate new x values for the fitted function and calculate the corresponding y values
x_fit = np.linspace(min(x_data), max(x_data), num=100)
y_fit = fitted_function(x_fit)

# Plot the original data points and the fitted function
plt.scatter(x_data, y_data, label='Data Points', color='red')
plt.plot(x_fit, y_fit, label='Fitted Function', color='blue')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar

df2= pd.read_csv('preprocessed_p (1).csv', index_col = 'created_at', parse_dates = True)

train=df2.iloc[185000:196000,0]


# Function to find the root where y falls below the threshold
def find_x_for_threshold(threshold, x_data, y_data, degree):
    coefficients = np.polyfit(x_data, y_data, degree)
    print(f"Fitted Polynomial Coefficients: {coefficients}")

    def polynomial_function(x):
        y = 0
        for i, coeff in enumerate(coefficients[::-1]):
            y += coeff * x**i
        return y

    def target_function(x):
        return polynomial_function(x) - threshold

    result = root_scalar(target_function, method='ridder', bracket=[min(x_data), max(x_data)])
    return result.root

# Example usage:
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    x_data = np.arange(1, 11001)
    y_data = np.array(train)

    # Define the threshold value for y
    threshold = 20
    
    # Degree of the polynomial to fit
    degree = 2

    # Find the x value where y falls below the threshold
    x_threshold = find_x_for_threshold(threshold, x_data, y_data, degree)

    # Plot the data points and the fitted polynomial curve
    x_values = np.linspace(min(x_data), max(x_data), 100)
    y_values = np.polyval(np.polyfit(x_data, y_data, degree), x_values)

    plt.scatter(x_data, y_data, label='Data Points')
    plt.plot(x_values, y_values, color='r', label='Fitted Polynomial Curve')
    plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
    plt.scatter(x_threshold, threshold, color='g', label='Crossing Point')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Fitted Polynomial Curve and Threshold')
    plt.grid(True)
    plt.show()

    print(f"The y value falls below the threshold {threshold:.2f} at x = {x_threshold:.2f}")