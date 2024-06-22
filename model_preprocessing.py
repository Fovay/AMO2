import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)
    scaler = StandardScaler()
    data['temperature'] = scaler.fit_transform(data[['temperature']])
    data.to_csv(output_file, index=False)

os.makedirs('processed_train', exist_ok=True)
os.makedirs('processed_test', exist_ok=True)
preprocess_data('train/data_train.csv', 'processed_train/data_train.csv')
preprocess_data('test/data_test.csv', 'processed_test/data_test.csv')
