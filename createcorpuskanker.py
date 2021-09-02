# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:20:52 2021

@author: aluenen
"""

import json
import re

def openfile(krant):
    """
    Opens the JSON file of a given newspaper

    Args: The name of the newspaper of interest (str)
    Returns: The data in the txt file (str)
    """
    infile = 'D:/Newsdata/Original/' + krant + ' (print).json'
    f = open(infile, 'r', encoding='utf-8') #changed: encoding='utf-8'
    print('file opened')
    return f

def selecttxt(article):
    """
    Collects the text of a given article

    Args: a loaded line from a JSON file
    Returns: The text in that file (str)
    """
    text = ''
    try:
        text = text + ' ' + article['text']
    except:
        pass
    try:
        text = text + ' ' + article['byline']
    except:
        pass
    try: 
        text = text + ' ' + article['title']
    except:
        pass 
    return text

def selectdata(file):
    """
    Selects text from articles published after 2009 that contain the word "kanker"

    Args: The opened JSON file
    Returns: A list of strings where each string contains the text of one article (lst)
    """
    artkanker = []
    for line in file:
        article = json.loads(line)
        date = article["publication_date"]
        year = int(date[:4])
        if year > 2009:
            text = selecttxt(article)
            if "kanker" in text: #also run it with ' kanker ' to see the difference
                artkanker.append(re.sub("\n", '', text))
    print(len(artkanker))
    return artkanker

def writedata(data):
    """
    Writes given data to a file called corpuskanker.txt

    Args: a list of strings (lst)
    Returns: a file containing
    """
    with open("D:/Newsdata/Original/corpuskanker.txt", 'a', encoding='utf-8') as outfile:
        for string in data:
            outfile.write(string + '\n')
        
        
def main():
    print("Hello World!")
    papers = ['ad', 'nrc', 'trouw', 'telegraaf', 'volkskrant']  
    for paper in papers:
        print("now working on...", paper)
        file = openfile(paper)
        print("selecting data")
        data = selectdata(file)
        print("writing away data")
        writedata(data)
        

if __name__ == "__main__":
    main()        
