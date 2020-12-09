import numpy as np
import glob
import os
import pandas as pd

from tqdm import tqdm
import nltk
import string
from nltk.tokenize import word_tokenize
import random
import pickle
from nltk.corpus import stopwords

from autocorrect import Speller
import re
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer 

from hyperopt import fmin, tpe, hp

# load a document
def load(filename):
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()
    return text

# split a document into news story and highlights
def split(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories from a directory
def load_stories(directory):
    stories = []
    for name in os.listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load(filename)
        # split into story and highlights
        story, highlights = split(doc)
        # store
        stories.append({'story':story, 'highlights':highlights})
    return stories

directory = r'C:\Users\ymaha\Desktop\cnn\stories'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

def preprocesing(lines):
    
    # function to convert nltk tag to wordnet tag
    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def lemmatize_sentence(sentence):
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    #     print(wordnet_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    #         if tag is not None:
    #             lemmatized_sentence.append(lemmatizer.lemmatize(word, tag)) 

        return " ".join(lemmatized_sentence)   
    
    
    temp = []
    for line in lines:
        # strip source cnn 
        index = line.find('(CNN)')
        if index > -1:
            line = line[index+len('(CNN)'):]

        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation and special characters from each token
        line = [w.replace('[<>!#@$:.,%\?-_]+', ' ') for w in line]
        # remove non ascii characters
        line = [w.replace('[^\x00-\x7f]', ' ') for w in line]        
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
#         # removing stop words
#         line = [word for word in line if word not in stop_list]
        # removing words of length 1
        line = [word for word in line if len(word) > 1]   
#         # Lemmatizing the words and combing them into a line
#         temp.append(lemmatize_sentence(' '.join(line)))
        # Combining the words into a line
        temp.append(' '.join(line))
    # remove empty strings
    temp = [c for c in temp if len(c) > 0]
    return temp

stop_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

for i in tqdm(range(len(stories))):
# for example in stories:
    stories[i]['story'] = preprocesing(stories[i]['story'].split('\n'))
    stories[i]['highlights'] = preprocesing(stories[i]['highlights'])
    
# save to file
from pickle import dump
dump(stories, open('processed_cnn_data.pkl', 'wb'))


    
    