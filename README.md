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
(pending completion)

### 2. Feature Engineering and Selection
(pending)

### 3. Training and Testing Sets

The data was split into training and testing sets using sklearn's train_test_split function. The parameters selected include: 
- X: features (TBD)
- y: target (label true or fake)
- test_size: Since the dataset is large, standard set sizes were selected, 80% for training the model and 20% for testing. 
- random_state: the random_state parameter was included to ensure reproducible results

### 4. Model

**Model 1: Naive Bayes Classifier (NB)**
The NB model was selected because it is a classifier that uses probabilistic algorithms to generate predictions. It works well with data that originally consists of natural language and it is able to perform analysis efficiently. Another benefit of NB is that it is able to handle irrelevant features. One of the limitations of NB is that it assumes features are completely independent, so the relationship between features is lost, however for this analysis this limitation is acceptable because we are focusing on individual words rather than the combination of words. 

**Model 2: Support Vector Machine (SVM)**
The SVM model was selected because it is a binary classification model that is able to find the best line or hyperplane to determine if the data belongs to one of two classes. One of the benefits of SVM is the ability to select a kernel, since the data is (linearly separable or not linearly separable), the kernel selected is (linear or rbf). Another benefit of SVM is its ability to accomodate outliers. A limitation of SVM is that it can take time to train on larger datasets.