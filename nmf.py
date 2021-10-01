# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:35:55 2021

@author: aluenen
"""
#We should also look at this: https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb
#https://github.com/derekgreene/topic-model-tutorial

import joblib
import numpy as np
from sklearn import decomposition

class NonNegativeMatrixFactorisation:
    '''This class contains all functions to execute NMF'''
    
    def openjoblib(self, name):
        """
        Opens a given joblib file file
        
        Args: A path to a joblib file (str)
        Returns: A document/term matrix (A), a list of feature names (terms), \
            and a list of snippets of articles
        """
        (A,terms,snippets) = joblib.load(name)
        print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )
        return A, terms, snippets
    
    def decomposition(self, k, A):
        '''
        Runs topic model

        Parameters
        ----------
        k : number of topics (int)
        A : document/term matrix

        Returns
        -------
        W : document membership weights of each of the topics
        H : term weights to each of the topics

        '''
        model = decomposition.NMF( init="nndsvd", n_components=k ) 
        W = model.fit_transform( A )
        H = model.components_
        return W, H
    
    def get_descriptor(self, terms, H, topic_index, top):
        '''
        Extract top terms for each topic

        Parameters
        ----------
        terms : list of tokens
        H : term weights to each of the topics
        topic_index : index of topic in H
        top : number of terms per topic

        Returns
        -------
        top_terms : list of top terms per topic

        '''
        # reverse sort the values to sort the indices
        top_indices = np.argsort( H[topic_index,:] )[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            top_terms.append( terms[term_index] )
        return top_terms
    
    def printtopics(self, k, H, terms):
        descriptors = []
        for topic_index in range(k):
            descriptors.append(self.get_descriptor( terms, H, topic_index, 10))
            str_descriptor = ", ".join( descriptors[topic_index] )
            print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )
            
    def savemodel(self, W, H, terms, snippets ):
        joblib.dump((W,H,terms,snippets), "articles-model-nmf-k%02d.pkl" % k)

def main():
    nmf = NonNegativeMatrixFactorisation()
    k = 12 #number of topics
    A, terms, snippets = nmf.openjoblib("D:/Newsdata/Original/corpuskanker-tfidf.pkl")
    W, H = nmf.decomposition(k, A)
    nmf.printtopics(k, H, terms)
    #nmf.savemodel(W, H, terms, snippets)
    
    


if __name__ == "__main__":
    main()        
