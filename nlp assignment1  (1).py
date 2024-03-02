#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from collections import defaultdict


# In[2]:


#step 1+2+3 : Word tokenization, Token Normalaization and Vocabulary Set Extraction
def extract_vocabulary(folder_path):
    vocabulary_set = set()
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()

    class_documents = defaultdict(list)

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                document_content = file.read()
                #Word Tokenization
                tokens = word_tokenize(document_content)

                #Token Normalization
                normalized_tokens = [porter_stemmer.stem(token.lower()) for token in tokens
                                     if token.isalpha() and token.lower() not in stop_words]

                #Vocabulary Set Extraction
                vocabulary_set.update(normalized_tokens)
                class_name = root.split(os.path.sep)[-1]
                class_documents[class_name].append(normalized_tokens)

    return vocabulary_set, class_documents


# In[3]:


def estimate_model_parameter(vocabulary_set, class_documents):
    class_counts = defaultdict(int)
    word_counts_per_class = defaultdict(lambda: defaultdict(int))

    #Count occurrences of each word in each class
    for class_name, documents in class_documents.items():
        class_counts[class_name] += len(documents)
        for document_tokens in documents:
            for word in document_tokens:
                word_counts_per_class[class_name][word] += 1

    #Calculate prior probabilities
    total_instances = sum(class_counts.values())
    num_classes = len(class_counts)
    prior_probabilities = {class_name: (count + 1) / (total_instances + num_classes)
                          for class_name, count in class_counts.items()}

    # Calculate likelihood
    vocabulary_size = len(vocabulary_set)
    likelihood = defaultdict(lambda: defaultdict(float))
    for class_name, word_counts in word_counts_per_class.items():
        total_words_in_class = sum(word_counts.values())
        for word in vocabulary_set:
            likelihood[class_name][word] = (word_counts[word] + 1) / (total_words_in_class + vocabulary_size)

    return prior_probabilities, likelihood


# In[11]:


def classify_documents(document_tokens, prior_probabilities, likelihood, alpha=0.0001):
    class_scores = defaultdict(float)

    for class_, prior_prob in prior_probabilities.items():
        score = np.log(prior_prob)  # Use logarithm to avoid underflow
        for word in document_tokens:
            smoothed_likelihood = (likelihood[class_name].get(word, 0) + alpha) / (sum(likelihood[class_name].values()) + alpha * len(likelihood[class_name]))
            score += np.log(smoothed_likelihood)
        class_scores[class_name] = score
        
    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class


# In[6]:


folder_path = 'C:\\Users\\dell\\Downloads\\training'

vocabulary_set, class_documents = extract_vocabulary(folder_path)
prior_probabilities, likelihood = estimate_model_parameter(vocabulary_set, class_documents)


# In[7]:


train_data = {}
test_data = {}

for class_name, documents in class_documents.items():
    if len(documents) >= 2:  
        train_documents, test_documents = train_test_split(documents, test_size=0.2)
        train_data[class_name] = train_documents
        test_data[class_name] = test_documents


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# Lists to store true and predicted labels
true_labels = []
predicted_labels = []
for class_name, test_documents in test_data.items():
    for document_tokens in test_documents:
        predicted_class = classify_documents(document_tokens, prior_probabilities, likelihood)
        true_labels.append(class_name)
        predicted_labels.append(predicted_class)

# Calculate precision, recall, F1-score, and support for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')

print(f"Micro-Averaged F1-Score: {f1_score:.2f}")


# In[ ]:




