#!/usr/bin/env python
# coding: utf-8

# # nlp project youssef shahen 20107033 Ahmed hesham

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# In[2]:


file_path = r'D:\books for study\nlp\final project\Restaurant_Reviews.tsv'
data = pd.read_csv(file_path, delimiter='\t', quoting=3)


# In[3]:


data.head()


# In[4]:


data.describe


# In[5]:


print("Missing values in the training set:")
print(data.isnull().sum())


# In[6]:


data['Liked'].value_counts()


# In[7]:


sns.countplot(x=data['Liked'])


# In[8]:


data['Review Letter Count']=data['Review'].apply(len)
data


# In[9]:


data.iloc[data['Review Letter Count'].idxmax()][0]


# In[10]:


data['Review'] =data['Review'].apply(lambda x: x.lower())
data.head()


# In[11]:


corpus=data['Review']


# In[12]:


import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['Review'] = data['Review'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
data.head()


# In[13]:


data['Review'] = data['Review'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
data.head()


# In[14]:


import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
data['Review'] = data['Review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
data.head()


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


cv =CountVectorizer()


# In[17]:


cv.fit_transform(corpus).toarray().shape


# In[18]:


x=cv.fit_transform(corpus).toarray()
x


# In[19]:


y=data['Liked']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# # using machine learning algorthim

# In[20]:


from sklearn.naive_bayes import MultinomialNB


# In[21]:


clf=MultinomialNB()


# In[22]:


clf.fit(x_train,y_train)


# In[23]:


y_pred=clf.predict(x_test)


# In[24]:


y_test.values


# In[25]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[26]:


print(confusion_matrix(y_test,y_pred))


# In[27]:


print(accuracy_score(y_test,y_pred))


# In[28]:


print(classification_report(y_test,y_pred))


# In[29]:


text = 'I Like This Restaurant'
text = text.lower()
text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
text = text.translate(str.maketrans('', '', string.punctuation))
text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
text_vectorized = cv.transform([text]).toarray()
prediction_nb = clf.predict(text_vectorized)
print(f"Predicted sentiment using Naive Bayes: {prediction_nb}")

if prediction_nb==1:
    print(f"Predicted sentiment using Naive Bayes: positive")
elif prediction_nb==0:
    print(f"Predicted sentiment using Naive Bayes: negative")


# In[30]:


text = 'I hate This Restaurant'
text = text.lower()
text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
text = text.translate(str.maketrans('', '', string.punctuation))
text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
text_vectorized = cv.transform([text]).toarray()
prediction_nb = clf.predict(text_vectorized)
print(f"Predicted sentiment using Naive Bayes: {prediction_nb}")

if prediction_nb==1:
    print(f"Predicted sentiment using Naive Bayes: positive")
elif prediction_nb==0:
    print(f"Predicted sentiment using Naive Bayes: negative")


# # using deep learning algorthim

# In[31]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[32]:


input_feature_dim = x_train.shape[1]
num_classes = 2
epochs = 10
batch_size = 32


# In[33]:


y_train_one_hot = tf.one_hot(y_train, depth=num_classes)
y_test_one_hot = tf.one_hot(y_test, depth=num_classes)


# In[34]:


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=input_feature_dim))

model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test_one_hot))

loss, accuracy = model.evaluate(x_test, y_test_one_hot)
print(f'Accuracy: {accuracy}')


# In[35]:


y_pred_prob = model.predict(x_test)
y_pred = tf.argmax(y_pred_prob, axis=1).numpy()

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[36]:


text = 'I Like This Restaurant'
text = text.lower()
text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
text = text.translate(str.maketrans('', '', string.punctuation))
text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
text_vectorized = cv.transform([text]).toarray()
text_pred_prob = model.predict(text_vectorized)
text_pred = tf.argmax(text_pred_prob, axis=1).numpy()
print(f"Predicted sentiment using Deep Learning: {text_pred}")

if prediction_nb==1:
    print(f"Predicted sentiment using Naive Bayes: positive")
elif prediction_nb==0:
    print(f"Predicted sentiment using Naive Bayes: negative")


# In[37]:


text = 'I hate this restaurant'
text = text.lower()
text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
text = text.translate(str.maketrans('', '', string.punctuation))
text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
text_vectorized = cv.transform([text]).toarray()
text_pred_prob = model.predict(text_vectorized)
text_pred = tf.argmax(text_pred_prob, axis=1).numpy()
print(f"Predicted sentiment using Deep Learning: {text_pred}")

if prediction_nb==1:
    print(f"Predicted sentiment using Naive Bayes: positive")
elif prediction_nb==0:
    print(f"Predicted sentiment using Naive Bayes: negative")


# In[ ]:




