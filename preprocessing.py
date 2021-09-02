# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:28:58 2021

@author: aluenen
"""
import nltk
import json
import re

def openfile(filename):
    """
    Opens a given file

    Args: A path to a txt file (str)
    Returns: A list of strings that are articles (lst)
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.readlines() 
    return data

def removenum(data):
    """
    Removes digits from the data

    Args: A list of strings that are articles (lst)
    Returns: A list of strings with digits removed (lst)
    """
    datawonum = []
    for article in data:
        artwonum = re.sub("\d", '', article) #match all digits
        datawonum.append(artwonum)
    return datawonum

def removepunct(datawonum):
    """
    Removes punctuation except hyphens between words

    Args: A list of strings with digit removed (lst)
    Returns: The list of strings with punctuation removed (lst)
    """
    datawopunct = []
    for article in datawonum:
        artwohyphen = re.sub("\W[-]\W", '', article) #match all hyphens that are not between words
        artwopunct = re.sub("[^\w\s-]", '', artwohyphen) #match all punctuation except hyphens
        datawopunct.append(artwopunct)
    return datawopunct

def lowercase(datawopunct):
    """
    Lowercases the data

    Args: A list of strings with punctuation removed (lst)
    Returns: The list of strings with all characters lowercased(str)
    """
    datalower = []
    for article in datawopunct:
        artlower = article.lower()
        datalower.append(artlower)
    return datalower
    
def tokenize(datawopunct):
    """
    Tokenizes the data based on whitespace

    Args: A list of strings with punctuation removed (lst)
    Returns: A list of lists of tokens (lst)
    """
    tokens = []
    for article in datawopunct:
        arttokens = article.split()
        tokens.append(arttokens)
    return tokens

def removestop(tokens):
    """
    Removes stopwords and one-character tokens

    Args: A list of lists of tokens (lst)
    Returns: A list of lists of tokens but without stopwords and one-character tokens (lst)
    """
    stopword_list1 = nltk.corpus.stopwords.words('dutch')
    datawostops = []
    for article in tokens:
        artwoshorts = [t for t in article if len(t) > 2]
        artwostops = [w for w in artwoshorts if w not in stopword_list1]
        datawostops.append(artwostops)
    return datawostops

def writedata(datawostops):
    """
    Dumps the data to a JSON file called

    Args: a list of strings (lst)
    Returns: a file containing the preprocessed
    """
    with open("D:/Newsdata/Original/corpuskanker_processed.json", 'w', \
              encoding='utf-8') as outfile:
        json.dump(datawostops, outfile)

    
def main():
    print("opening file")
    file = openfile("D:/Newsdata/Original/corpuskanker.txt")
    print("remove numbers")
    datawonum = removenum(file)
    print('remove punctuation')
    datawopunct = removepunct(datawonum)
    print('lowercasing')
    datalower = lowercase(datawopunct)
    print('tokenizing')
    tokens = tokenize(datalower)
    print('removing stop words')
    datawostops = removestop(tokens)
    print("dumping data to 'corpuskanker_processed.json'")
    writedata(datawostops)
    print("finished")
    
        

if __name__ == "__main__":
    main()        

