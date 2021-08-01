# Import Dependencies
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Read Test Data into a Dataframe
root = os.path.dirname(os.path.abspath(__file__))  
test_file_path = os.path.join(root, 'test_data_multiple.csv')
article_df = pd.read_csv(test_file_path)

# Natural Language Processing
article_df['article'] = article_df['title']+" "+article_df['text']
article_df['title'] = article_df['title'].str.replace('U.S.', 'USA').str.replace('U.S', 'USA').str.replace(' US ', ' USA ')
article_df['text'] = article_df['text'].str.replace('U.S.', 'USA').str.replace('U.S', 'USA').str.replace(' US ', ' USA ')
article_df['article'] = article_df['article'].str.replace('U.S.', 'USA').str.replace('U.S', 'USA').str.replace(' US ', ' USA ')
@np.vectorize
def wordpre(x):
    x = x.lower()
    x = re.sub('(?<!\w)([A-Za-z])\.', r'\1', x)
    x = re.sub('“|’|"|”', '', x)
    x = re.sub('\[.*?\]', '', x)
    x = re.sub("\\W"," ",x)
    x = re.sub('https?://\S+|www\.\S+', '', x)
    x = re.sub('<.*?>+', '', x)
    x = re.sub('[%s]' % re.escape(string.punctuation), '', x)
    x = re.sub('\n', '', x)
    x = re.sub('\w*\d\w*', '', x)
    return x
article_df['title']= article_df['title'].apply(wordpre)
article_df['text']= article_df['text'].apply(wordpre)
article_df['article']= article_df['article'].apply(wordpre)
stop = stopwords.words('english')
article_df['title']= article_df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
article_df['text']= article_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
article_df['title_count'] = article_df['title'].apply(len)-1
article_df['text_count'] = article_df['text'].apply(len)-1
article_df['title_tokens'] = article_df['title'].apply(word_tokenize)
article_df['text_tokens'] = article_df['text'].apply(word_tokenize)
article_df['title_tokenized_count'] = article_df['title_tokens'].apply(len)
article_df['text_tokenized_count'] = article_df['text_tokens'].apply(len)
subject_dummies = pd.get_dummies(article_df['subject'])
ndf = pd.DataFrame()
text = article_df['article'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize)
for i in text: 
    N = nltk.pos_tag(i)
    C = Counter([j for i,j in N])
    S = pd.Series([C])
    N = pd.DataFrame.from_records(S, columns = S.sum().keys())
    ndf = pd.concat([ndf, N], ignore_index=True, sort=False)
ndf["sum"] = ndf.sum(axis=1)
ndf = ndf.div(ndf['sum'], axis=0) *100
features_df = pd.concat([
    article_df.drop(['subject','title','text','article','title_tokens','text_tokens'],axis=1),
    subject_dummies,
    ndf.drop('sum',axis=1)
    ], axis=1)

# Handle Missing Required Features
reqd_features = ['title_count', 'text_count', 'title_tokenized_count',
    'text_tokenized_count', 'US News', 'World News', 'JJ', 'NN', 'VBZ',
    'RP', 'VBG', 'VBP', 'DT', 'RB', 'VB', 'CC', 'PRP', 'IN', 'VBD', 'TO',
    'PRP$', 'NNS', 'JJS', 'CD', 'JJR', 'RBR', 'VBN', 'MD', 'WP', 'FW',
    'NNP', 'WRB', 'WDT', 'PDT', 'EX', 'RBS', 'NNPS', 'UH', 'WP$', 'POS']
for i in reqd_features:
    if(i not in list(features_df.columns)):
        features_df[i]=0
features_df = features_df.fillna(0)

# Scale Features
scaler_file_path = os.path.join(root,'../static/machineLearning/scaler.sav')
scaler = pickle.load(open(scaler_file_path,'rb'))
scaled_df = pd.DataFrame(scaler.transform(features_df),columns=features_df.columns)

# Predict Outcome
svm_file_path = os.path.join(root,'../static/machineLearning/svm_model.sav')
svm_model = pickle.load(open(svm_file_path,'rb'))
result = svm_model.predict(scaled_df)

# Print Result
print(f"The outcome is :{result}")