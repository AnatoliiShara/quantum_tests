import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler

class RegressionModel:
    def __init__(self, train_file):
        self.train_data = pd.read_csv(train_file)
        self.X = self.train_data.drop(columns=['target'])
        self.y = self.train_data['target']
        self.lgb_model = None
    
    def train_lightgbm(self):
        train_data = lgb.Dataset(self.X, label=self.y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
        }
        self.lgb_model = lgb.train(params, train_data, num_boost_round=100)
    
    def predict_lightgbm(self, test_file):
        test_data = pd.read_csv(test_file)
        test_X = test_data  # Assuming the test data has the same columns as training data
        predictions = self.lgb_model.predict(test_X)
        return predictions

if __name__ == "__main__":
    # Initialize the RegressionModel and train LightGBM model
    model = RegressionModel('train.csv')
    model.train_lightgbm()

    # Predict values from 'hidden_test.csv'
    predictions = model.predict_lightgbm('hidden_test.csv')

    # Save the predictions to a CSV file
    output_df = pd.DataFrame({'predictions': predictions})
    output_df.to_csv('predictions.csv', index=False)
