# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Gather and preprocess the dataset, ensuring features (e.g., house size, location) and target variables (price, occupants) are cleaned, scaled, and split into training/testing sets.

2. Model Initialization: Initialize two separate SGD Regressor models or a multi-output SGD Regressor to handle multiple targets.

3. Model Training: Train the SGD Regressor(s) on the training dataset by iteratively updating weights to minimize prediction errors.

4. Model Evaluation: Assess the model’s performance using MSE for both targets (price and occupants) on the test dataset.

5. Prediction: Input new house feature data to predict both price and number of occupants using the trained model.


## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SARANYA S.
RegisterNumber: 212223220101


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Load the california Housing 
data = fetch_california_housing()
#df =pd.DataFrame(data.data,columns=data.feature_names)
#df['target']=data.target
#print(df.head())

#use the first 3 features as inputs
x=data.data[:,:3] #featurs:'MEdInc','Houseage','AceRooms'

#use 'MeHouseVal' and 'AveOccup' as output variables
y=np.column_stack((data.target,data.data[:,6])) #targets: 'MeHouseVal' 'AveOccup'

#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

#initialize the SGDRegressor

sgd = SGDRegressor(max_iter=1000,tol=1e-3)

#use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

#train the model
multi_output_sgd.fit(x_train,y_train)

#predict on the test data
y_pred = multi_output_sgd.predict(x_test)

#Inverse transform the predictions to get them back to the original scale
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

#Evaluate the model using Mean squared error
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

#optionally,print some predictions
print("\nPredictions:\n",y_pred[:5]) #print first 5 predictions
```

## Output:
X value:

![image](https://github.com/user-attachments/assets/aba170bb-95c2-496f-aed5-232ef40cbb5a)

 Y value:
 
 ![image](https://github.com/user-attachments/assets/980ae798-067d-4d75-b1e1-c287257186b6)
 
Mean squared value and predicted value:

![image](https://github.com/user-attachments/assets/9ddb4be8-2d60-4e9f-adc1-f16e680fd0f3)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
