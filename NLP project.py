#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dropout
nltk.download('punkt')
nltk.download('stopwords')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


stemmer = PorterStemmer()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)


# In[4]:


def preprocess_dataset(data):
    preprocessed_data = []
    for text in data:
        preprocessed_text = preprocess_text(text)
        preprocessed_data.append(preprocessed_text)
    return preprocessed_data


# In[5]:


def read_dataset(dataset_path):
    classes = sorted(os.listdir(dataset_path))
    data = []
    labels = []

    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(class_path):
            with open(os.path.join(class_path, file_name), 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
                labels.append(label)

    return data, labels, classes


# In[6]:


def read_and_preprocess_dataset(dataset_path):
    data, labels, classes = read_dataset(dataset_path)
    preprocessed_data = preprocess_dataset(data)
    return preprocessed_data, labels, classes


# In[7]:


dataset_path = 'C:\\Users\\dell\\Downloads\\training'
preprocessed_data, labels, classes = read_and_preprocess_dataset(dataset_path)


# In[8]:


def extract_vocabulary(preprocessed_data):
    vectorizer = CountVectorizer()
    vectorizer.fit(preprocessed_data)
    vocabulary_set = vectorizer.vocabulary_.keys()
    return vocabulary_set


# In[9]:


vocabulary_set = extract_vocabulary(preprocessed_data)


# In[10]:


def tfidf_encoding(preprocessed_data):
    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(preprocessed_data)
    return tfidf_features, vectorizer

tfidf_features, vectorizer = tfidf_encoding(preprocessed_data)


# In[23]:


X_train, X_val, y_train, y_val = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_val_preds = nb_model.predict(X_val)
nb_result = classification_report(y_val, nb_val_preds, output_dict=True)

accuracy = nb_result['accuracy']
f_score = nb_result['macro avg']['f1-score']

print("Naive Bayes Classification:")
print("Accuracy:", accuracy)
print("F1-score:", f_score)


# In[48]:


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_val_preds = svm_model.predict(X_val)

svm_result = classification_report(y_val, svm_val_preds, output_dict=True)

accuracy = svm_result['accuracy']
f_score = svm_result['macro avg']['f1-score']

print("SVM Classification:")
print("Accuracy:", accuracy)
print("Macro Average F1-score:", f_score)


# In[49]:


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_val_preds = rf_model.predict(X_val)
rf_result = classification_report(y_val, rf_val_preds, output_dict=True)

accuracy = rf_result['accuracy']
f_score = rf_result['macro avg']['f1-score']

print("Random Forest Classification:")
print("Accuracy:", accuracy)
print("Macro Average F1-score:", f_score)


# In[17]:


from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow.keras.initializers import Constant
import gensim.downloader as api
from gensim.models import KeyedVectors

word2vec_model = api.load('word2vec-google-news-300')

glove_file = 'C:\\Users\\dell\\Downloads\\glove.6B\\glove.6B.300d.txt'
word2vec_output_file = 'C:\\Users\\dell\\Downloads\\glove.6B\\glove.6B.300d.word2vec.txt'
glove2word2vec(glove_file, word2vec_output_file)
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

def create_embedding_matrix(embedding_model, tokenizer, embedding_dim):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in embedding_model.wv.vocab:
            embedding_matrix[i] = embedding_model[word]
    return embedding_matrix


# In[ ]:


from scipy.sparse import csr_matrix

X_train_array = X_train.toarray()
X_val_array = X_val.toarray()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)

embedding_dim = 300 
vocab_size = len(tokenizer.word_index) + 1 

word2vec_embedding_matrix = create_embedding_matrix(word2vec_model, tokenizer, embedding_dim)
glove_embedding_matrix = create_embedding_matrix(glove_model, tokenizer, embedding_dim)

word2vec_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
              embeddings_initializer=Constant(word2vec_embedding_matrix),
              trainable=False),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100),
    Dropout(0.2),
    Dense(len(classes), activation='softmax')
])

word2vec_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

glove_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
              embeddings_initializer=Constant(glove_embedding_matrix),
              trainable=False),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100),
    Dropout(0.2),
    Dense(len(classes), activation='softmax')
])

glove_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

word2vec_model.fit(X_train_array, y_train, validation_data=(X_val_array, y_val), epochs=10, batch_size=64)
glove_model.fit(X_train_array, y_train, validation_data=(X_val_array, y_val), epochs=10, batch_size=64)


# In[ ]:


word2vec_scores = word2vec_model.evaluate(X_val, y_val, verbose=0)
word2vec_predictions = np.argmax(word2vec_model.predict(X_val), axis=-1)
word2vec_result = classification_report(y_val, word2vec_predictions, output_dict=True)

word2vec_accuracy = word2vec_scores['accuracy']
word2vec_f_score = word2vec_result['macro avg']['f1-score']

print("Word2Vec Model Evaluation:")
print("Accuracy:", word2vec_accuracy)
print("Macro Average F1-score:", word2vec_f_score)

glove_scores = glove_model.evaluate(X_val, y_val, verbose=0)
glove_predictions = np.argmax(glove_model.predict(X_val), axis=-1)
glove_result = classification_report(y_val, glove_predictions, output_dict=True)

glove_accuracy = glove_scores['accuracy']
glove_f_score = glove_result['macro avg']['f1-score']

print("\nGloVe Model Evaluation:")
print("Accuracy:", glove_accuracy)
print("Macro Average F1-score:", glove_f_score)


# In[ ]:





# In[ ]:


#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, LSTM, Dense

#from tensorflow.keras.layers import Dropout

#X_dense = tfidf_features.toarray()

#X_train, X_val, y_train, y_val = train_test_split(X_dense, labels, test_size=0.2, random_state=42)

#y_train = np.array(y_train)
#y_val = np.array(y_val)

#embedding_dim = 100
#lstm_model = Sequential([
 #   Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim),
  #  LSTM(units=100, return_sequences=True),
   # Dropout(0.2),
    #LSTM(units=100),
    #Dropout(0.2),
    #Dense(len(classes), activation='softmax')
#])
#lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
#lstm_val_preds = np.argmax(lstm_model.predict(X_val), axis=-1)
#print("\nLSTM Classification Report:")
#print(classification_report(y_val, lstm_val_preds))


# In[ ]:





# In[ ]:




