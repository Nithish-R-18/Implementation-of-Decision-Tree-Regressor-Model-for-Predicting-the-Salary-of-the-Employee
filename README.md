# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries, load the Salary dataset, and convert categorical columns such as Position into numerical values using Label Encoding.

2. Select Position and Level as input features (X) and Salary as the target variable (y). Split the dataset into training and testing sets.

3. Create the DecisionTreeRegressor model and train it using the training dataset.

4. Predict salary values for the test dataset, calculate the R² score to measure performance, and visualize the decision tree using plot_tree

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NITHISH R
RegisterNumber: 212225230202 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

X = df[['Level']]
y = df['Salary']

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()


```

## Output:

<img width="935" height="77" alt="ml ex 9 1" src="https://github.com/user-attachments/assets/3123b1ae-689b-4219-92ad-94b01ce3413f" />


<img width="793" height="576" alt="ml ex 9 2" src="https://github.com/user-attachments/assets/e6a90d46-57b7-48e0-aefb-9ed50a58cd41" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
