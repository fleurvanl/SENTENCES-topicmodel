# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:28:58 2021

@author: aluenen
"""
import nltk
import json
import re
import operator
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessing:
    """This class contains all functions for preprocessing"""

    def openfile(self, filename):
        """
        Opens a given file
        
        Args: A path to a txt file (str)
        Returns: A list of strings that are articles (lst)
        """
        with open(filename, 'r', encoding='utf-8') as file:
            data = file.readlines() 
        return data

    def removenum(self, data):
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

    def removepunct(self, datawonum):
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

    def lowercase(self, datawopunct):
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
    
    def tokenize(self, datawopunct):
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

    def removestop(self, tokens):
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

    #def writedata(self, datawostops):
    #    """
    #    DOESN'T WORK WITH THE DEREK GREENE TUTORIAL!
    #    Dumps the data to a JSON file called
    #    
    #    Args: a list of strings (lst)
    #    Returns: a file containing the preprocessed
    #    """
    #    with open("D:/Newsdata/Original/corpuskanker_processed_TEST.json", 'w', \
    #              encoding='utf-8') as outfile:
    #        json.dump(datawostops, outfile)
            
    def joinarticle(self, datawostops):
        """
        Joins a list of lists of strings into a list of strings
        
        Args: A list of lists of lists of strings (lst)
        Returns: A list of strings where each string consists of the tokens in one article(lst)
        """
        articles = []
        for article in datawostops:
            articles.append(' '.join(article))
        return articles
    
    def getsnippets(self, articles):
        """
        Obtains the up to 100 first tokens of an article to serve as a title
        
        Args: A list of lists of strings where the strings are the tokens of an article(lst)
        Returns: A list of strings where each string consists of the first 100 tokens in one article(lst)
        """
        snippets = []
        for article in articles:
            snippets.append(article[0:min(len(article), 100)])
        return snippets
            
    def rank_terms(self, A, terms):
        # get the sums over each column
        sums = A.sum(axis=0)
        # map weights to the terms
        weights = {}
        for col, term in enumerate(terms):
            weights[term] = sums[0,col]
        # rank the terms by their weight over all documents
        return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

    def createtfidf(self, datawostops, snippets):
        """
        Creates a document matrix based on the term-frequency inverse-document-freq
        
        Args: a list of strings (lst)
        Returns: a Joblib dump .pkl file
        """
        vectorizer = TfidfVectorizer(min_df = 20)
        A = vectorizer.fit_transform(datawostops)
        print( "Created %d X %d TF-IDF-normalized document-term matrix" % (A.shape[0], A.shape[1]) )    
        terms = vectorizer.get_feature_names()
        print("Vocabulary has %d distinct terms" % len(terms))
        ranking = self.rank_terms(A, terms)
        for i, pair in enumerate( ranking[0:20] ):
            print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ) )
        joblib.dump((A, terms, snippets), "D:/Newsdata/Original/corpuskanker-tfidf.pkl")

def main():
    pp = Preprocessing()
    print("opening file")
    file = pp.openfile("D:/Newsdata/Original/corpuskanker.txt")
    print("remove numbers")
    datawonum = pp.removenum(file)
    print('remove punctuation')
    datawopunct = pp.removepunct(datawonum)
    print('lowercasing')
    datalower = pp.lowercase(datawopunct)
    print('tokenizing')
    tokens = pp.tokenize(datalower)
    print('removing stop words')
    datawostops = pp.removestop(tokens)
    print("joining articles to one str again")
    articles = pp.joinarticle(datawostops)
    print("getting snippets")
    snippets = pp.getsnippets(articles)
    print("getting tf-idf")
    pp.createtfidf(articles, snippets)
    #print("dumping data to 'corpuskanker_processed_TEST.json'")
    #pp.writedata(datawostops)
    print("finished")
    
        

if __name__ == "__main__":
    main()        

