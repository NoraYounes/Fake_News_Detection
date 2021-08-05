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
6. Merge the True and Fake data into one DataFrame

### 2. Feature Engineering and Selection
The dataset data type consist of natural language, thus several Natural Language Processing (NLP) methods were applied to generate features with numerical values that can be used in the machine learning model. 

**1. Converting to Lowercase, Removing Punctuation and Stopwords:**
An additional preprocessing step was taken to help reduce noise in the data to help in the NLP process and so that features focus on the words. 
- Replacing U.S and U.S. with USA in title and text data  
- Lowercase and Punctuation Removal: A function was created to convert the title and text data to lowercase and remove punctuation such as apostrophes. 
- StopWords were also removed for the title and text data using nltk's stopwords library. 

**2. Word Count:**
The word count for title and text will be used as a feature as the length may indicate whether its true or false. 

**3. Tokenized Count:**
Tokenization was applied on the title and text data to gain a different type of insight through the words being split into meaningful units. The features would consist of the number of tokens per title and title.

**4. Part-of-Speech Tagging (POS-tag) & Normalization:** 
POS-tag was used on the text data to tag words into different types/tags such as adjectives, adverbs, preposition, nouns and verbs. The sum of each tag in an article will be used as a feature based on the assumption that the type of word can predict whether an article is true or fake. For example, fake articles may use more verbs than true articles. The POS-tag features were normalized to a scale between 0-100. 

**5. Encoding Categorical Variables:** 
The subject column was encoded using the get_dummies() function where US News and World News were turned into indicator columns. 

**6. CountVectorizer:**
The CountVectorizer was used on the text data to generate a feature which consists of the occurence and tokenization of each word per article. 


#### Summary of Features:
1. **Word Count:**
- Title Word Count
- Text Word Count

2. **Tokenized Count:**
- Title Tokenized Count
- Text Tokenized Count 

3. **Subject:**
- US News 
- World News 

4. **POS-tag:**
- CC (coordinating conjunction)
- CD (cardinal digit)
- DT (determiner)
- EX (existential there (like: “there is” … think of it like “there exists”))
- FW (foreign word)
- IN (preposition/subordinating conjunction)
- JJ (adjective)
- JJR (adjective, comparative)
- JJS (adjective, superlative)
- LS (list item marker)
- MD (modal)
- NN (noun, singular)
- NNS (noun plural)
- NNP (proper noun, singular)
- NNPS (proper noun, plural)
- PDT (predeterminer)
- POS (possessive ending)
- PRP (personal pronoun)
- PRP$ (possessive pronoun)
- RB (adverb)
- RBR (adverb, comparative)
- RBS (adverb, superlative)
- RP (particle)
- TO (‘to’)
- UH (interjection)
- VB (verb, base form)
- VBD (verb, past tense)
- VBG (verb, gerund/present participle)
- VBN (verb, past participle)
- VBP (verb, non-3rd person singular present)
- VBZ (verb, 3rd person singular present)
- WDT (wh-determiner)
- WP (wh-pronoun)
- WP$ (possessive wh-pronoun)
- WRB (wh-adverb)

5. **CountVectorizer**
- Vectorized Word Count 

### 3. Training and Testing Sets
The data was split into training and testing sets using sklearn's train_test_split function. The parameters selected include: 
- X: Summary of Features (listed above, not including the Vectorized Word Count)
- y: Label
- test_size: Since the dataset is large, standard set sizes were selected, 80% for training the model and 20% for testing. 
- random_state: the random_state parameter was included to ensure reproducible results

### 4. Machine Learning Model

**Model 1: Naive Bayes Classifier (NB)**
The NB model was selected because it is a classifier that uses probabilistic algorithms to generate predictions. It works well with data that originally consists of natural language and it is able to perform analysis efficiently. Another benefit of NB is that it is able to handle irrelevant features. One of the limitations of NB is that it assumes features are completely independent, so the relationship between features is lost, however for this analysis this limitation is acceptable because we are focusing on individual words rather than the combination of words. 

**Model 2: Support Vector Machine (SVM)**
The SVM model was selected because it is a binary classification model that is able to find the best line or hyperplane to determine if the data belongs to one of two classes. One of the benefits of SVM is its ability to accomodate outliers. A limitation of SVM is that it can take time to train on larger datasets.

### 5. Model Training
**Model 1: Naive Bayes Classifier (NB)**
Two different NB models were trained using different types of NB algorithms:

1. **GaussianNB (gb_model)**
The gb_model was trained using the GaussianNB algorithm. 

2. **MultinomialNB (nb_model)**
The nb_model only used the Vectorized Word Count as the X features. The results of the CountVectorizer on the text data generates a significant number of features/columns, so in order to run the nb_model, 10% of the articles dataset was used due to processing limitations. The nb_model was trained using the MultinomialNB algorithm. 

**Model 2: Support Vector Machine (SVM)**
The svm_model was trained using the linear kernel. 

### 6. Model Validation 
1. **GaussianNB (gb_model)**
The  image below shows the gb_model validation results. The accuracy was 88.8%. The confusion matrix shows that the model had 3837 True Positives (true article as true), 2853 True Negatives (fake articles as fake), 431 False Negatives (true articles as fake) and 410 False Positives (fake articles as true). The f1-score further supports the performance of the model with 0.90 for True classification and 0.87 for Fake classification. 

<<img width="546" alt="gb_validation" src="https://user-images.githubusercontent.com/78664640/128411468-384d8d8e-6b89-4f36-929d-657986190147.png">

2. **MultinomialNB (nb_model)**
The  image below shows the gb_model validation results. The accuracy was 95.9%. The confusion matrix shows that the model had 387 True Positives (true article as true), 335 True Negatives (fake articles as fake), 11 False Negatives (true articles as fake) and 20 False Positives (fake articles as true). The f1-score further supports the performance of the model with 0.96 for both True and False classification. 

![nb_validation](https://user-images.githubusercontent.com/78664640/128411470-875a6bf8-ddea-4d85-a211-95e44478c32c.png)

3. **Support Vector Machine (svm_model)**
The  image below shows the SVM model validation results. The accuracy was 92.5%. The confusion matrix shows that the model had 4068 True Positives (true article as true), 2898 True Negatives (fake articles as fake), 200 False Negatives (true articles as fake) and 365 False Positives (fake articles as true). The f1-score further supports the performance of the model with 0.94 for True classification and 0.91 for Fake classification. 

![svm_validation](https://user-images.githubusercontent.com/78664640/128411473-2dee3398-2e54-445a-b8db-dae2b09ac3d7.png)

### 7. Model Selection
The gb_model and svm_model used the engineered features (excluding the vectorized word count) and scaling, while the nb_model only used the vectorized word count as a feature. During the model integration into the dashboard, both the gb_model and svm_model resulted in issues due to the regex applied during the NLP process ans scaling in the machine learning testing, thus the final model selected for the dashboard was the nb_model. Ultimately, the nb_model had the highest accuracy and easiest implementation into the dashboard. 

> ## Dashboard 
[Click here to check out our Dashboard](http://fakefactdetector.herokuapp.com/)


> ## Presentation
[Click here to check out our Presentation](https://docs.google.com/presentation/d/1oDIxY25KyXxs1QhZghPIj5Z90Xhdd0Rf4pUN0oJYf6U/edit?usp=sharing)