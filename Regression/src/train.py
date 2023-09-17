import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training data
train_data = pd.read_csv('train.csv')

# Split the data into features (X) and the target variable (y)
X = train_data.drop(columns=['target'])
y = train_data['target']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset for training
train_dataset = lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_valid, label=y_valid)

# Define LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
}

# Train the LightGBM model 
num_round = 100
bst = lgb.train(
    params,
    train_dataset,
    num_round,
    valid_sets=[valid_dataset]  
)

# Save the trained model to a file
bst.save_model('trained_model.txt')