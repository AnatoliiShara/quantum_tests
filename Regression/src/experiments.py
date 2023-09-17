import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import numpy as np

class RegressionModel:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.X = self.data.drop(columns=['target'])
        self.y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
    
    def train_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        self.linear_reg = LinearRegression()
        self.linear_reg.fit(self.X_train, self.y_train)
    
    def train_lightgbm(self):
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
        }
        self.lgb_model = lgb.train(params, train_data, num_boost_round=100)
    
    def evaluate_linear_regression(self):
        linear_reg_predictions = self.linear_reg.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, linear_reg_predictions))
        mae = mean_absolute_error(self.y_test, linear_reg_predictions)
        r2 = r2_score(self.y_test, linear_reg_predictions)
        return rmse, mae, r2
    
    def evaluate_lightgbm(self):
        lgb_predictions = self.lgb_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, lgb_predictions))
        mae = mean_absolute_error(self.y_test, lgb_predictions)
        r2 = r2_score(self.y_test, lgb_predictions)
        return rmse, mae, r2

if __name__ == "__main__":
    model = RegressionModel('train.csv')
    
    # Train and evaluate Linear Regression
    model.train_linear_regression()
    linear_reg_rmse, linear_reg_mae, linear_reg_r2 = model.evaluate_linear_regression()
    print("Linear Regression RMSE:", linear_reg_rmse)
    print("Linear Regression MAE:", linear_reg_mae)
    print("Linear Regression R-squared:", linear_reg_r2)
    
    # Train and evaluate LightGBM
    model.train_lightgbm()
    lgb_rmse, lgb_mae, lgb_r2 = model.evaluate_lightgbm()
    print("LightGBM RMSE:", lgb_rmse)
    print("LightGBM MAE:", lgb_mae)
    print("LightGBM R-squared:", lgb_r2)
