# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:28:24 2021

@author: aluenen
"""

import joblib
import numpy as np
from nmf import NonNegativeMatrixFactorisation
nmf = NonNegativeMatrixFactorisation()
from sklearn import decomposition
from itertools import combinations
from gensim.models.word2vec import Word2Vec

class TC_W2V:
    '''this class contains all functions to optimize a topic model using TC-W2V'''
    
    def nmffork(self, kmin, kmax, A):
        topicmodels = []
        for k in range(kmin,kmax+1):
            print("Applying NMF for k=%d ..." % k )
            model = decomposition.NMF( init="nndsvd", n_components=k ) 
            W = model.fit_transform( A )
            H = model.components_    
            topicmodels.append( (k,W,H) )
        return topicmodels
    
    def loadw2v(self, path):
        return Word2Vec.load(path)
    
    def calculate_coherence(self, w2v_model, term_rankings ):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # check each pair of terms
            pair_scores = []
            for pair in combinations( term_rankings[topic_index], 2 ):
                try: #try except for KeyError(word not in vocabulary)
                    pair_scores.append( w2v_model.wv.similarity(pair[0], pair[1]) )
                except:
                    pass
                    #print(pair)
            # get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # get the mean score across all topics
        return overall_coherence / len(term_rankings)
    
    def get_descriptor(self, all_terms, H, topic_index, top):
        # reverse sort the values to sort the indices
        top_indices = np.argsort( H[topic_index,:] )[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            top_terms.append( all_terms[term_index] )
        return top_terms
    
    def compare_models(self, topic_models, terms, w2v_model):
        k_values = []
        coherences = []
        for (k,W,H) in topic_models:
            # Get all of the topic descriptors - the term_rankings, based on top 10 terms
            term_rankings = []
            for topic_index in range(k):
                term_rankings.append(self.get_descriptor(terms, H, topic_index, 10))
        # Now calculate the coherence based on our Word2vec model
            k_values.append(k)
            coherences.append(self.calculate_coherence(w2v_model, term_rankings))
            print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )
    
    
    
    
 
def main():
    tc = TC_W2V()
    kmin = 5
    kmax = 25
    A, terms, snippets = nmf.openjoblib("D:/Newsdata/Original/corpuskanker-tfidf.pkl")
    topicmodels = tc.nmffork(kmin, kmax, A)   
    w2vmodel = tc.loadw2v("D:/Newsdata/Models/uncased/300_10_15.model")
    tc.compare_models(topicmodels, terms, w2vmodel)
    
if __name__ == "__main__":
    main()        