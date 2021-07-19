# Fake News Detection 🔎

![WhatsApp Image 2021-07-08 at 10 31 36 AM](https://user-images.githubusercontent.com/78935551/125214201-b1696680-e283-11eb-870c-b1433543ce50.jpeg)

> ## Overview
Fake news is false or misleading information presented as news. It often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. There is no fixed definition of a false news story, and, in fact, the term is used more broadly to encompass any type of misinformation, including unintentional and unconscious mechanisms, and also to apply to news unfavourable to one's own personal views.

In a time when we can spread and receive information quickly, thanks to the technology and 4G networks , media literacy is very important. 
Detecting fake news has become more important than ever for the sake of a peaceful society. A prediction tool would be a terrific way to help people make more educated decisions on news. 

We are working on a solution that users can use to identify news containing false and misleading information. The post and title of a fake article are analyzed using simple and carefully selected features. 

> ## Data Sourcing 
A large set of True and Fake news would be used to create a model which would predict whether an input is True or False based on that model.

- The primary Dataset is from [University of Victoria](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php) and [Kaggle Dataset](https://www.kaggle.com/c/fake-news/data).
- [Top 20 Fake/Real words](https://www.kdnuggets.com/2017/04/machine-learning-fake-news-accuracy.html) using Naive Bayes classifier.

> ## The purpose of this analysis is to answer the following question:
- Is the article true or false? 
- What is the accuracy of the predictive model? 
- Fake news has many sources, where do most of them come from?

> ## Communication Protocols

The communication methods selected for this project include:

1. **Slack:** primary channel for messaging during meetings and is used to share resources/project files
2. **Zoom:** video conferencing platform used for team meetings
3. **Google Docs:** used to document meeting notes and project planning elements
4. **Whatsapp:** alternative channel used to schedule team meetings and is used for messaging when the team is offline
 

> ## Machine Learning Model

### 1. Data Preprocessing
The primary dataset from the University of Victoria includes two csv files, a Fake.csv that consists of articles deemed Fake and a True.csv for True articles. Each dataset has 4 columns which include the article title, text, subject and date. The Fake dataset has 23,481 records with 6 types of subjects (News, left-news, politics, Government News, Middle-east and US_News) and the True dataset has 21,417 records with 2 types of subjects (politicsNews and worldnews). After the initial data exploration, the following steps were taken to preprocess the data:

1. Delete articles with blank 'text' values
2. Drop duplicate records
3. Organize 'subjects' into two categories, US News and World News
- In order to keep the subjects consistent, we reduced the 6 Fake subjects into the 2 categories, to match the True subjects and renamed them. 
- The Fake data has a lower number of World News (742) than US News (16,712), compared to the True data with US News (11,202) and World News at (9,989). At this stage, we decided that 742 records is sufficient and to keep the data as is because the primary goal is to predict Fake vs. True articles. 
4. Drop the 'date' column: 
- Since both the true and fake article records were collected from 2016 to 2017, we decided to eliminate the date column to focus on more insightful data. 
5. Add label columns with a value of '0' for True and '1' for Fake
6. Merge the True and Fake data into a one DataFrame

### 2. Feature Engineering and Selection
The dataset data type consist of natural language, thus several Natural Language Processing (NLP) methods were applied to generate numerical features that can be used in the machine learning model. 

1. Converting to Lowercase, Removing Punctuation and Stopwords:
An additional preprocessing step was taken to help reduce noise in the data to help in the NLP process and so that features focus on the words. A function was created to convert the title and text data to lowercase and remove punctuation such as apostrophes. StopWords such as 'the' were also removed for the title and text data using nltk's stopwords library. 

2. Word Count
The word count for title, text and article (title+text) will be used as a feature as the length of the article may indicate whether its true or false. 

3. Tokenized Count
Tokenization was applied on the title, text and article to gain a different type of insight through the words being split into meaningful units. The feature would consist of the number of tokens per title, text and article. 

4. **Part-of-Speech Tagging (POS-tag)**: 
POS-tag was used on the 'text' column to tag words into different types/tags such as adjectives, adverbs, preposition, nouns and verbs. The sum of each tag in an article will be used as a feature based on the assumption that the type of word can predict whether an article is true or fake. For example, fake articles may use more verbs than true articles. 

5. **Encoding Categorical Variables**: 
The subject column was encoded using the get_dummies() function where US News = 1 and World News = 0. 


#### Summary of Features:
- Title Word Count
- Text Word Count
- Article Word Count
- Title Tokenized Count
- Text Tokenized Count 
- Article Tokenized Count
- Subject
- Label (True vs. False)
- Adjectives
- Adverbs
- Preposition
- Nouns
- Verbs 

### 3. Training and Testing Sets
The data was split into training and testing sets using sklearn's train_test_split function. The parameters selected include: 
- X: Features (not including the Label)
- y: Label
- test_size: Since the dataset is large, standard set sizes were selected, 80% for training the model and 20% for testing. 
- random_state: the random_state parameter was included to ensure reproducible results

### 4. Model

**Model 1: Naive Bayes Classifier (NB)**
The NB model was selected because it is a classifier that uses probabilistic algorithms to generate predictions. It works well with data that originally consists of natural language and it is able to perform analysis efficiently. Another benefit of NB is that it is able to handle irrelevant features. One of the limitations of NB is that it assumes features are completely independent, so the relationship between features is lost, however for this analysis this limitation is acceptable because we are focusing on individual words rather than the combination of words. 

**Model 2: Support Vector Machine (SVM)**
The SVM model was selected because it is a binary classification model that is able to find the best line or hyperplane to determine if the data belongs to one of two classes. One of the benefits of SVM is the ability to select a kernel, since the data is (linearly separable or not linearly separable), the kernel selected is (linear or rbf). Another benefit of SVM is its ability to accomodate outliers. A limitation of SVM is that it can take time to train on larger datasets.