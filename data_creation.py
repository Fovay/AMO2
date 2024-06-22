import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_data(num_days):
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(num_days)]
    temperature = np.random.normal(0, 1, num_days).cumsum()
    return pd.DataFrame({'date': dates, 'temperature': temperature})
train_data = generate_data(365)
test_data = generate_data(100)
train_data.to_csv('train/data_train.csv', index=False)
test_data.to_csv('test/data_test.csv', index=False)
