"""
SSL 2019
Project: LSTM Language Model Perplexity - Text Generation
Name: Tim Day
Program: A Simple Wikipedia Crawler to build an English language text corpus
         Only usng wiki article summaries
         Create a training set of at least 2000 lines 
         and a test set of at least 500 lines
"""

import wikipedia
from nltk import sent_tokenize
wikipedia.set_lang("en")
line_count = 0

with open("corpus_wiki/raw_wiki_corpus.txt", "a") as text_file:
    while line_count < 2500:
      try:
        random_wiki = wikipedia.random(pages=1)
        summary_text = wikipedia.summary(random_wiki)
        sentences = sent_tokenize(summary_text)
        # print(summary_text)
        # print(sentences)

        # TODO remove lines < 20 characters
        # TODO remove non-english chars?
        for sen in sentences:
          sen.lower()
          line_count += 1
          text_file.write(sen + '\n')
      except:
        print('ERROR: move on. Likely wikipedia disambiguation error (multiple pages with same title)')
        pass
        # just skip these pages
         

       