# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:21:52 2021

@author: aluenen
"""

#topicvisualisation
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import numpy as np

from gensim.models.word2vec import Word2Vec

path = "D:/Newsdata/Models/uncased/300_10_15.model"
model = Word2Vec.load(path)

topics = {1:["wel", "leven", "moeder", "mensen", "heel", "vader", "weer", "kinderen", "goed", "jaar"], \
          2:["patiënten", "patiënt", "artsen", "behandeling", "euthanasie", "arts", "medicijnen", "medicijn", "huisarts", "middel"],\
              3:["onderzoek", "dna", "onderzoekers", "cellen", "mensen", "eten", "wetenschap", "kanker", "wetenschappers", "vlees"],\
                  4:["ziekenhuizen", "zorg", "ziekenhuis", "zorgverzekeraars", "verzekeraars", "kwaliteit", "operaties", "patiënten", "behandelingen", "verzekeraar"],\
                      5:["the", "film", "overleden", "jarige", "jaar", "acteur", "regisseur", "jaren", "films", "leeftijd"],\
                          6:["armstrong", "lance", "tour", "doping", "usada", "livestrong", "uci", "france", "wielrennen", "renner"],\
                              7:["chávez", "president", "venezuela", "maduro", "hugo", "venezolaanse", "cuba", "caracas", "capriles", "oppositie"],\
                              8:["alpe", "euro", "kwf", "geld", "miljoen", "veenendaal", "kankerbestrijding", "deelnemers"],\
                                  9:["vrouwen", "borstkanker", "mannen", "procent", "jaar", "test", "bevolkingsonderzoek", "baarmoederhalskanker", "kans", "hpv"],\
                                      10:["asbest", "bedrijf", "euro", "volgens", "nederland", "jaar", "zegt", "bedrijven", "gemeente", "minister"],\
                                          11:["cruijff", "ajax", "johan", "barcelona", "feyenoord", "club", "trainer", "voetbal", "spelers", "wedstrijd"],\
                                              12:["roken", "rokers", "sigaret", "sigaretten", "tabaksindustrie", "longkanker", "tabak", "stoppen", "roker", "procent"]}
#"dhuzes", , "dhuez" excluded from topic 8 bc not in vocab 
        
my_vocab = {}
for t in topics.keys():
    for w in topics[t]:
        try:
            my_vocab[w] = model.wv[w]
        except KeyError:
            print("Word %s missing from vocabulary" %(w))
            
my_words = my_vocab.keys()

#print(my_vocab.keys())
        
df = pd.DataFrame.from_dict(my_vocab, orient='index')
    
x = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print(principalComponents.shape)
#print(principalComponents)

words_pc = dict(zip(my_words, principalComponents))
print(words_pc)
   
#X = principalComponents[:,0]#first column
#Y = principalComponents[:,1]#second column


for t in topics.keys():
    for num in range(len(topics[t])):
        word = topics[t][num]
        if num == 0:
            topic_pc = np.array([words_pc[word]])
        else:
            pc = [words_pc[word]]
            topic_pc = np.append(topic_pc, pc, axis=0)
    X = topic_pc[:,0]
    Y = topic_pc[:,1]
    plt.scatter(X, Y, label=topics[t][0])
    
plt.title("Words per topic visualised in the embedding space") 
plt.ylabel("PCA component 0") 
#plt.x_ticks(201000, 201900, 100)
plt.xlabel("PCA component 1") 

#add legend
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))



#adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()
