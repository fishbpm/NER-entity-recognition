# -*- coding: utf-8 -*-
"""
Author: Mark Fisher - Student 17073334
Task 3 of CE807 Assignment Two

This module trains the classifier using the training set,
and then tests the trained model on the wiki gold test set.

PLEASE NOTE this file is currently configured to use a small 2000-sentence sample of the training
corpus that I have already pickled to file,  just to show that the code is operational
This pickled training file  wiki_test.pkl  is provided in the submission folder
It was not possible to supply the whole pickled training set as it is much too large for Faser to handle.

TO RUN THE CLASSIFIER
PLEASE ENSURE that both >
     > wiki_test.pkl    (training set)
 AND > wikigold.conll.txt  (test data)   are in your local directory
 
The wikigold.conll.txt file has been provided in the submission folder for your convenience
If the code is run, the test set will then be built directly from file, prior to running the classifier

PLEASE NOTE this will get a much lower F-Score (obviously) because it is only using around 2% of the training set
It is for demonstration purposes only - to show that the code is operational.
"""
#import numpy as np

from crf_NER_feature_extraction import extract_features, build_training_set

from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score

import pickle as pkl

import nltk
from nltk import re
import en_core_web_sm

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics   


train_file = "aij-wikiner-en-wp2"
test_file = "wikigold.conll.txt"
NEW_TEST = True #set to True if you wish to re-build the test set
NEW_TRAIN = False #set to True if you wish to re-build the training set  
RARE = 1000 #max frequecny for a word to be considered rare

"""==========================build training set===========================================================
   ================ (else extract from file if already pickled) =========================================="""
if NEW_TRAIN:
    X_train, y_train = build_training_set(train_file)
else:
    training = pkl.load(open('wiki_train.pkl', 'rb'))
    X_train = list(training['features'])
    y_train = list(training['labels'])

          
"""==========================build testing sentences=======================================================
   ================ (else extract from file if already pickled) =========================================="""
if NEW_TEST:
    file = open(test_file, "r", encoding='utf8')
    nlp = en_core_web_sm.load()
    
    data = file.read()
    sents = re.split("(?:\\\\\\n){2}", data) #get each sentence
    uniques = [[],[]]
    #for each sentence in the test file
    for s, sent in enumerate(sents):#[:20]):
        words = re.split("\n", sent)
        tokens = [[],[],[]]
        #for each word in this sentence
        for word in words:
            #clean each word of extraneous garbage meta-tags (if any)
            word = re.sub("(\\\\'\w{2})+((\\\\\w{3})?\\\\\w{4}\s?)?","",word)
            #if there is no residual sensible string after cleanup, then abort this word
            if len(word.split()) > 1:
                #else get the text and label
                token, label = re.sub("\\\\$","",word).strip().split()
                #and append the token
                if len(re.sub("-","",token)) > 0: #provided it is not solely a punctation token
                    #then strip any hyphens to prevent spacy from tokenizing, before appending
                    tokens[0].append(re.sub("-","",token)) 
                else:
                    tokens[0].append(token)
                #append the label, remove any I/B prefixes (as these are redundant)
                tokens[1].append(re.sub("^.-","",label)) 
                #and keep original hyphenated token (where applicable)
                tokens[2].append(token) 
    
        #rebuild the sentence and pass it the NLP parser for POS tagging
        parsed = nlp(" ".join(tokens[0]))
        parsed_tokens = [[token.text, token.tag_] for token in parsed if not token.text.isspace()]
        
        #spacy occasionally splits tokens - most often when a token has a concluding period (.)
        #the following checks for any surplus tokens arising and re-attaches these back to the preceding token
        assert len(parsed_tokens) >= len(tokens[0])
        if len(parsed_tokens) > len(tokens[0]):
            position = 0
            for token in parsed:
                if not token.text.isspace():
                    if tokens[0][position] not in [token.text, parsed_tokens[position][0]]:
                        #concatenate the NEXT token
                        parsed_tokens[position][0] = str(parsed_tokens[position][0]) + str(parsed_tokens[position+1][0])
                        #the first tag is retained - only the token itself is concatenated
                        del parsed_tokens[position+1]
                        #confirm that we have re-established the original token, as originally sourced from the test file
                        assert parsed_tokens[position][0] == tokens[0][position]
                        position -= 1
                    position += 1
            #!!! check that we have successfully re-instated the original sentence
            assert len(parsed_tokens) == len(tokens[0])
             
        #initialise required shape for the sentence container
        sents[s] = parsed_tokens 
        quote = False
        
        #for each word in the sentence
        for t, token in enumerate(parsed_tokens):
            #check whether it is within quotations
            if quote:
                quote = (token[1] != "''") #close currently open quote, as tagged by spacy
            else:
                quote = (token[1] == '``') #open a new quote, as tagged by spacy
            #load this word into the sentence, along with its POS, label and quotation
            sents[s][t] = tuple([tokens[2][t], token[1], tokens[1][t], quote])
            #increment frequency of this word
            try:
                pos = uniques[0].index(tokens[2][t]) #find position in word array
            except:
                uniques[0].append(tokens[2][t]) #if not found then append the new word
                uniques[1].append(1) #and initialise with frequency of 1
            else:
                uniques[1][pos] += 1 #else increment the frequency
                
    """==========================build features and labels======================================================="""  
    X_test = []
    y_test = []
    for sent in sents:#[:20]:
        if len(sent) > 1:
            #first get the word frequencies (ie. do this once only for this sentence)
            frequency = [uniques[1][uniques[0].index(token[0])] for token in sent]
            #extract the features for each word in this sentence
            X_test.append([extract_features(sent, position, frequency) for position in range(len(sent))])
            #append the testing labels for each word in this sentence
            y_test.append([token[2] for token in sent])
    
    #Save dataset as pickle
    with open('wiki_test.pkl', 'wb') as f:
        pkl.dump({ 'features': X_test, 'labels': y_test }, f, pkl.HIGHEST_PROTOCOL)
else:
    testing = pkl.load(open('wiki_test.pkl', 'rb'))
    X_test = list(testing['features'])
    y_test = list(testing['labels'])


"""========================== classify =======================================================""" 

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(X_test)

print("Full Test Accuracy:", crf.score(X_test, y_test))
print("Full Test F1 Score:", metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=crf.classes_))
print("Trimmed Test F1 Score:", metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))


"""========================== append prior predictions and re-classify ========================"""
for sent, labels in enumerate(y_train):
    for word, label in enumerate(labels):
        if word > 0:
            X_train[sent][word]['prev.ent'] = labels[word - 1] #add prediction for previous word
        if word > 1:
            X_train[sent][word]['prev.ents'] = labels[word - 1] + labels[word - 2]
            
for sent, labels in enumerate(y_pred):
    for word, label in enumerate(labels):
        if word > 0:
            X_test[sent][word]['prev.ent'] = labels[word - 1] #add prediction for previous word
        if word > 1:
            X_test[sent][word]['prev.ents'] = labels[word - 1] + labels[word - 2]
            
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(X_test)

print("Full Test Accuracy:", crf.score(X_test, y_test))
print("Full Test F1 Score:", metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=crf.classes_))
print("Trimmed Test F1 Score:", metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))