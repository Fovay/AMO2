import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

test_data = pd.read_csv('processed_test/data_test.csv')
X_test = test_data[['temperature']]
y_test = test_data.index
model = joblib.load('model.pkl')

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')
