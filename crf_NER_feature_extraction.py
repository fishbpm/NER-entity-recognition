# -*- coding: utf-8 -*-
"""
Author: Mark Fisher - Student 17073334
Task 2 of CE807 Assignment Two

This module extracts all features from the WP2 training file, and pickles them to file.
As specified in the assignment remit for Task 2, this only covers the TRAINING set
A 2000-sentence SAMPLE of the pickled file has already been provided in TASK 3 of the submission folder.
It was not possible to supply the whole pickled training set as it is much too large for Faser to handle

So it is not necessary to run this file, unless you wish to.
a TRUNCATED version the wp2 file has been provided in the submission folder, in case you wish to do so.

"""
#import numpy as np

import time
import pickle as pkl

import nltk
from nltk import re
import en_core_web_sm


PICKLING = True
RARE = 1000 #max frequecny for a word to be considered rare
            #this is currently set to a high value as it turned out not to improve the F-score

"""==========================extract features ===========================================================
   PLEASE NOTE this module currently extracts ALL extended features for the model.
   This is to ensure you are able to clearly see the full set of features that were evaluated
   -------------
   HOWEVER, this full set of features does NOT produce the highest F-Score
   A SUBSET of these features (as indicated in Task 2 of the Report) produce the highest score
   This SUBSET was used to create the training set that was provided in the submission folder"""
def extract_features(sent, position, frequency):
    token = sent[position][0]
    tag = sent[position][1]
    quote = sent[position][3]
    
    #core features for all words
    features = {
    'bias': 1.0,
    'token': token.lower(),
    'pos.tag': tag,
    'pos.root': tag[:2],
    'quote': quote
    }
    
    #remaining features apply to content (low frequency) words only
    if frequency[position] < RARE:
        features.update({
        'suffix.major': token[-3:],
        'suffix.minor': token[-2:],
        'upper.case': token.isupper(),
        'title.case': token.istitle(),
        'length': min(15, len(token)),
        'numeric': token.isdigit(),
        'has.period': token.find('.') > -1,
        'has.hyphen': token.find('-') > -1
        })
    
    #repeat for next neighbouring token (select features only)
    if position > 0:
        token1 = sent[position-1][0]
        tag1 = sent[position-1][1]
        
        features.update({
        'prev.token': token1.lower(),
        'prev.tag': tag1,
        'prev.root': tag1[:2]
        })
        #remaining features apply to content (low frequency) words only
        if frequency[position-1] < RARE:
            features.update({
            'prev.title': token1.istitle(),
            'prev.upper': token1.isupper(),
            })          
    else:
        features['BOS'] = True #beginning of sentence
    
    #repeat for previous neighbouring token (select features only)
    if position < len(sent)-1:
        token1 = sent[position+1][0]
        pstag1 = sent[position+1][1]
        
        features.update({
        'next.token': token1.lower(),
        'next.tag': pstag1,
        'next.root': pstag1[:2]
        })
        #remaining features apply to content (low frequency) words only
        if frequency[position+1] < RARE:
            features.update({
            'next.title': token1.istitle(),
            'next.upper': token1.isupper(),
            })
    else:
        features['EOS'] = True #end of sentence

    return features

"""==========================build training sentences======================================================="""
def build_training_set(train_file):
    t0 = time.time()
    
    nlp = en_core_web_sm.load()
    file = open(train_file, "r", encoding='utf8')
    uniques = [[],[]] #frequency array for each unique word in the corpus
    
    sents = [line for line in file if line != '\n']
    del sents[0] #remove unwanted extraneous carriage return at top of input file
    #for each sentence in the training file
    for s, sent in enumerate(sents[:2000]):
        #rebuild sentence and pass it the NLP parser for POS tagging
        parsed = nlp(re.sub("(\|\S*){2}(\s|$)"," ",sent))
        #initialise sentence container
        sents[s] = re.sub('\\n+','',sent).split(" ")
        quote = False #some sentences fail to close open quotes. This forces closure
        for t, (token, tagged) in enumerate(zip(sents[s], parsed)):
            word = token.split("|")
            if quote:
                quote = (tagged.tag_ != "''") #close currently open quote, as tagged by spacy
            else:
                quote = (tagged.tag_ == '``') #open a new quote, as tagged by spacy
            sents[s][t] = tuple([word[0], tagged.tag_, re.sub("^.-","",word[2]), quote]) #remove I/B prefixes
            #increment frequency of this word
            try:
                pos = uniques[0].index(word[0]) #find position in word array
            except:
                uniques[0].append(word[0]) #if not found then append the new word
                uniques[1].append(1) #and initialise with frequency of 1
            else:
                uniques[1][pos] += 1 #else increment frequency
        if s/2000 == int(s/2000): #track the time (this is a long process)
            print(time.time() - t0, 'secs to proc', s, 'sentences')
    
    """==========================build features and labels======================================================="""
    X_train = []
    y_train = []
    for sent in sents[:2000]:
        if len(sent) > 1:
            #first get the word frequencies (ie. do this once only for this sentence)
            frequency = [uniques[1][uniques[0].index(token[0])] for token in sent]
            #extract the features for each word in this sentence
            X_train.append([extract_features(sent, position, frequency) for position in range(len(sent))])
            #append the training labels for each word in this sentence
            y_train.append([token[2] for token in sent])
    
    return X_train, y_train
          

if __name__ == '__main__':

    X_train, y_train = build_training_set("aij-wikiner-en-wp2")
    if PICKLING:    
        #Save the dataset, so we dont need to repeat all the above unnecessarily!
        with open('wiki_train.pkl', 'wb') as f:
            pkl.dump({ 'features': X_train, 'labels': y_train }, f, pkl.HIGHEST_PROTOCOL)