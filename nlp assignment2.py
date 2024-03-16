#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.models
import gensim.downloader as api
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

glove_input_file = 'C:\\Users\\dell\\Downloads\\glove.6B\\glove.6B.300d.txt'
word2vec_output_file = 'C:\\Users\\dell\\Downloads\\glove.6B\\glove.6B.300d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
glove2word2vec(glove_input_file, word2vec_output_file)


# In[2]:


glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
word2vec_model = api.load('word2vec-google-news-300')
fasttext_model = api.load('fasttext-wiki-news-subwords-300')


# In[3]:


file_path = 'C:\\Users\\dell\\Downloads\\train_rand_split.jsonl'
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))


# In[4]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

for item in data:
    item['question']['stem'] = clean_text(item['question']['stem'])
    for choice in item['question']['choices']:
        choice['text'] = clean_text(choice['text'])


# In[5]:


def text_to_vector(text, model):
    words = text.split()
    vectors = np.array([model.get_vector(word) for word in words if word in model])
    if len(vectors) == 0:
        return np.zeros(model.vector_size) 
    return np.mean(vectors, axis=0)


# In[6]:


def predict_answer(question, choices, model):
    question_vec = text_to_vector(question, model)
    best_choice = None
    max_similarity = -1
    for choice in choices:
        choice_text = choice['text']
        choice_vec = text_to_vector(choice_text, model)
        similarity = cosine_similarity([question_vec], [choice_vec])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_choice = choice['label']
    return best_choice


# In[7]:


def evaluate_accuracy(data, model):
    correct_predictions = 0
    for item in data:
        question = item['question']['stem']
        choices = item['question']['choices']
        correct_answer = item['answerKey']
        predicted_answer = predict_answer(question, choices, model)
        if predicted_answer == correct_answer:
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    return accuracy


# In[12]:


glove_accuracy = evaluate_accuracy(data, glove_model)
word2vec_accuracy = evaluate_accuracy(data, word2vec_model)
fasttext_accuracy = evaluate_accuracy(data, fasttext_model)

baseline_accuracy = 0.2
def difference(model_accuracy):
    return model_accuracy - baseline_accuracy

print(f"GloVe Model: {glove_accuracy}")
print(f"Improvement over baseline: {difference(glove_accuracy)*100:.2f}%")
print(f"Word2Vec Model: {word2vec_accuracy}")
print(f"Improvement over baseline: {difference(word2vec_accuracy)*100:.2f}%")
print(f"FastText Model: {fasttext_accuracy}")
print(f"Improvement over baseline: {difference(fasttext_accuracy)*100:.2f}%")


# In[15]:


#EXTRA: Testing some data from external file

test_data = []
with open('C:\\Users\\dell\\Downloads\\test.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        test_data.append(json.loads(line))

for item in test_data:
    question = item['question']['stem']
    choices = item['question']['choices']
    correct_answer = item['answerKey']
    predicted_answer_label = predict_answer(question, choices, glove_model)
    predicted_answer_text = next((choice['text'] for choice in choices if choice['label'] == predicted_answer_label), None)
    print(f"Question: {question}")
    print(f"Predicted Answer Label: {predicted_answer_label}, Predicted Answer: {predicted_answer_text}")
    correct_answer_text = next((choice['text'] for choice in choices if choice['label'] == correct_answer), None)
    print(f"Correct Answer: {correct_answer_text}\n")

test_accuracy = evaluate_accuracy(test_data, word2vec_model)  
print(f"word2vec_model accuracy: {test_accuracy}")


# In[ ]:




