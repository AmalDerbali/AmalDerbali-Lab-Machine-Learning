import joblib
import pandas as pd

#ref: 
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://stackoverflow.com/questions/48947194/add-randomforestclassifier-predict-proba-results-to-original-dataframe

prediction_model = joblib.load('Prediction_model')
data= pd.read_csv('test_samples.csv')

probability=prediction_model.predict_proba(data)
prediction= prediction_model.predict(data)

prob= pd.DataFrame(probability, 
                       columns= ['probability class 0', 'probability class 1'])
pred= pd.DataFrame(prediction,
                        columns=['prediction'])


df=pd.concat([prob, pred], axis=1)
df.to_csv('prediction.csv')

