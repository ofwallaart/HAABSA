#!/usr/bin/env python
# encoding: utf-8

import numpy as np
np.set_printoptions(threshold=np.inf)
import nltk
import time
from nltk import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def main(train_path, test_path, accuracyOnt, test_size, remaining_size, c=0.1, gamma=0.00001):
    punctuation_and_numbers = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', '\'', '’','*', '$', '0', '1', '2', '3',
                                   '4', '5', '6', '7', '8', '9']
    sentence_vector , review_vector, polarity_vector, cat_vector = [],[],[],[]
    # Load Train File
    with open(train_path, "r") as fd:
        lines = fd.read().splitlines()
        len_train = int(len(lines)/4)
        for i in range(0, len(lines), 4):
            #polarity
            polarity_vector.append(lines[i + 3].strip().split()[0])

            #load Category
            cat = lines[i+1]
            cat_vector.append(cat)

            #load target and sentence
            words = lines[i].lower()
            words_review = lines[i + 2].lower()

            #Remove punctuation
            for _ in punctuation_and_numbers:
                words_review = words_review.replace(_, '')
            for _ in punctuation_and_numbers:
                words = words.replace(_, '')
            
            review_vector.append(words_review)
            sentence_vector.append(words)
    # Load Test File
    with open(test_path, "r") as fd:
        lines = fd.read().splitlines()
        len_test = int(len(lines)/4)
        for i in range(0, len(lines), 4):
            #polarity
            polarity_vector.append(lines[i + 3].strip().split()[0])

            #load Category
            cat = lines[i+1]
            cat_vector.append(cat)

            #load target and sentence
            words = lines[i].lower()
            words_review = lines[i + 2].lower()

            #Remove punctuation
            for _ in punctuation_and_numbers:
                words_review = words_review.replace(_, '')
            for _ in punctuation_and_numbers:
                words = words.replace(_, '')
            
            review_vector.append(words_review)
            sentence_vector.append(words)

    sentence_vector = np.array(sentence_vector)

    cat_vector = np.array(cat_vector)
    review_vector = np.array(review_vector)

    polarity_vector = np.array(polarity_vector)

    # Construct the feature vector
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(cat_vector)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    one_hot_cat = np.array(onehot_encoded)

    cvec = CountVectorizer(lowercase=True, binary=True)
    wm = cvec.fit_transform(review_vector)
    bow = np.array(wm.A)

    # Get sentiment sentence score
    sent_score = []
    sid = SentimentIntensityAnalyzer()
    for sentence in sentence_vector:	
         ss = sid.polarity_scores(sentence)
         sent_score.append(ss['compound'])

    # Expend feature vector by sentiment score
    sent_score = np.expand_dims(np.array(sent_score), axis=1)

    # merge categories with sentiment score
    features1 = np.append(onehot_encoded, sent_score, axis =1)
    # merge bow with other features
    features_final = np.append(bow, features1, axis=1)

    train, test = np.split(features_final, [len_train])
    train_pol, test_pol = np.split(polarity_vector, [len_train])

    # SVM Machine
    svm_model_linear = SVC(kernel = 'linear', C = c, gamma = gamma).fit(train, train_pol)
    svm_in_predictions = svm_model_linear.predict(train)
    svm_predictions = svm_model_linear.predict(test)
     
    # model accuracy for X_test 
    in_accuracy = svm_model_linear.score(train, train_pol)
    accuracy = svm_model_linear.score(test, test_pol)
    totalacc = ((accuracy * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
    print('train acc = {:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(in_accuracy, accuracy, totalacc))
    return accuracy