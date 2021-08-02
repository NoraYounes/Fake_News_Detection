# Import Dependencies
import pandas as pd
import os
import pickle

# Read Test Data into a Dataframe
root = os.path.dirname(os.path.abspath(__file__))  
test_file_path = os.path.join(root,'test_data_single.csv')
article_df = pd.read_csv(test_file_path)

# Data Pre-processing
cv_file_path = os.path.join(root,'../static/machineLearning/cv.sav')
cv = pickle.load(open(cv_file_path,'rb'))
cv_text = cv.transform(article_df['text'])

# Predict Outcome
nb_file_path = os.path.join(root,'../static/machineLearning/nb_model.sav')
nb_model = pickle.load(open(nb_file_path,'rb'))
result = nb_model.predict(cv_text)

# Print Result
print(f"The outcome is :{result}")