import pandas as pd
import lightgbm as lgb
import numpy as np

# Load the trained LightGBM model
bst = lgb.Booster(model_file='trained_model.txt')

# Load the hidden test data
test_data = pd.read_csv('hidden_test.csv')

# Make predictions using the trained model
test_predictions = bst.predict(test_data)

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({'target': test_predictions})

# Save the predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)