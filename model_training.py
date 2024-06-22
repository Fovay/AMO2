import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

train_data = pd.read_csv('processed_train/data_train.csv')
X_train = train_data[['temperature']]
y_train = train_data.index

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')
