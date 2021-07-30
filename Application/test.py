import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from flask import Flask, render_template
from config import postgrespwd
import re
import string
from collections import Counter
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

articleInfo = {
    'subject':'US News',
    'title':"'I am furious with myself': Unvaccinated Covid patient describes the exhausting illness",
    'text':'Sitting in her hospital room in Baton Rouge, Louisiana, Aimee Matzen struggled to breathe as she described how exhausting it is to have Covid-19.Vaccine requirements are on the rise as Delta bariants The US is returning to early pandemic surges and restrictions. It&#39;s time to compel people to do the right thing, expert saysVaccine requirements are on the rise as Delta bariants The US is returning to early pandemic surges and restrictions. Its time to compel people to do the right thing, expert saysThe fact that I am here now, I am furious with myself," she told CNN between deep, deliberate breaths. "Because I was not vaccinated.Matzen, 44, finds herself in the Covid-19 intensive care unit at Our Lady of the Lake Regional Medical Center in Baton Rouge. She is receiving oxygen treatments and hopes she stays well enough to avoid getting hooked up to a ventilator.With Covid-19 surging in states across the country, Louisiana stands among those hardest hit by the most recent rise in cases, driven in large part by the Delta variant.The state has the highest 7-day average of new cases per-capita in the country, at 77 cases reported per 100,000 residents each day over the past week, according to a CNN analysis of data from Johns Hopkins University.It is a kick in the gut to feel like we effectively have lost six or seven months of progress," Louisiana State Health Officer Dr. Joseph Kanter told CNNs John King on Wednesday.Aimee Matzen, 44, is in the Covid-19 ICU at Our Lady of the Lake Regional Medical Center in Baton Rouge.Aimee Matzen, 44, is in the Covid-19 ICU at Our Lady of the Lake Regional Medical Center in Baton Rouge.Kanter attributed the surge to a "perfect storm" of factors, including the Delta variant, which is believed to be more transmissible, and "unacceptably low vaccination coverage.Louisianas vaccination rate is among the lowest in the country, with just 37% of residents fully vaccinated as of Wednesday, according to data from the US Centers for Disease Control and Prevention. Its the fifth lowest in the country, and Louisiana is one of six states that has less than 38% of residents fully vaccinated.The states largest healthcare system, Ochsner, has seen a 700% increase in Covid-19 patients over the last month and a 75% increase in the last week, officials said during a news conference on Wednesday.And the vast majority of those patients -- 88%, according to Ochsner Health CEO Warner Thomas -- are unvaccinated.This is absolutely disproportionately hitting folks that are unvaccinated,Thomas said. Those are the folks that in a very high majority wereseeing coming to the hospital.Matzen told CNN she was not opposed to getting vaccinated -- she just hadnt gotten around to it. Every time she planned to get inoculated, "something would come up," she said.I have this feeling ... if I was vaccinated, I wouldntbe hospitalized," Matzen sai'
    }
article_df = pd.DataFrame(articleInfo, index=[0])
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
reqd_features = ['title_count', 'text_count', 'title_tokenized_count',
    'text_tokenized_count', 'US News', 'World News', 'JJ', 'NN', 'VBZ',
    'RP', 'VBG', 'VBP', 'DT', 'RB', 'VB', 'CC', 'PRP', 'IN', 'VBD', 'TO',
    'PRP$', 'NNS', 'JJS', 'CD', 'JJR', 'RBR', 'VBN', 'MD', 'WP', 'FW',
    'NNP', 'WRB', 'WDT', 'PDT', 'EX', 'RBS', 'NNPS', 'UH', 'WP$', 'POS']
for i in reqd_features:
    if(i not in list(features_df.columns)):
        features_df[i]=0
root = os.path.dirname(os.path.abspath(__file__))  
ml_file_path = os.path.join(root, 'static/svm_model.sav')
model = pickle.load(open(ml_file_path, 'rb'))
result = model.predict(features_df)
print(f"The result is {result[0]}")
