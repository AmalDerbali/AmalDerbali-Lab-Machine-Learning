import joblib
import pandas 
from sklearn.preprocessing import StandardScaler

#ref: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

prediction_model = joblib.load('Prediction_model')
data= pandas.read_csv('test_samples.csv')

probability=prediction_model.predict_proba(data)
prediction= prediction_model.predict(data)

prediction_df= pandas.DataFrame(probability, prediction, 
                                columns=['probability class 0', 'probability class 1', 'prediction'])

prediction_df.to_csv('prediction.csv')