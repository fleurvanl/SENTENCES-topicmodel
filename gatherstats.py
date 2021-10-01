# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:15:04 2021

@author: aluenen
"""
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import Preprocessing
pp = Preprocessing()

with open('D:/Newsdata/Original/corpuskanker.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
    
df = pd.DataFrame({"body": data})
print(df.head())

def word_count(text):
    return len(str(text).split(' '))

def preprocess(text):
    text1 = pp.removepunct(text)
    text2 = pp.lowercase(text1)
    text3 = pp.tokenize(text2)
    return text3

def unique_words(text): 
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    return ulist

df['word_count'] = df['body'].apply(word_count) #apply means apply this function to every row in the given column
print(df['word_count'].mean())
    
#df['processed'] = df['body'].apply(preprocess)
#print(df.head())

print(df['word_count'].describe())

#get all words in ONE list
p_text = df['body']
p_text = [item for sublist in p_text for item in sublist]
num_unique_words = len(set(p_text))
print(num_unique_words)


# Plot a hist of the word counts
#fig = plt.figure(figsize=(10,5))

#plt.hist(
#    df['word_count'],
#    bins=50,
#    color='#60505C'
#)

#plt.title('Distribution - Article Word Count', fontsize=16)
#plt.ylabel('Frequency', fontsize=12)
#plt.xlabel('Word Count', fontsize=12)
#plt.yticks(np.arange(0, 50, 5))
#plt.xticks(np.arange(0, 2700, 200))

#plt.show()
    