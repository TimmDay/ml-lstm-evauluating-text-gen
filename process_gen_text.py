#  import generated text
import nltk
import re
import enchant
import matplotlib.pyplot as plt

print(enchant.list_languages())

d = enchant.Dict("en_GB") # change to en_US, en_GB, en_AU as required
file_corpus = 'corpus_tweets' # or corpus_wiki 
file_gen_text = ''


# print(d.check("Hello"))
# print(d.check("Helo"))
# print(d.suggest("Helo"))

### analyse corpus ###
# dictionary of all words/tokens in corpus
  # filter out punctuation groups of 3 or less chars
  # filter out numbers
  # filter out words with non-ascii chars
## zipf plot of corpus
# check all corpus words vs dictionary
# get corpus dictionary match rate



### analyse generated text ###
# go through all text. ignore lines starting with -, or a number, or 'Epoch'
  # should ignore seed text as well..
      # if line startswith (------- generating with seed:), 
      # update a variable to be array of seed. 
      # ignore words that match it until next update
# make a big list of all words in text/corpus

# select 100 sentences.

# filter out the seed words ( first x words as determined by algo)

# make a big list of all generated words
# check each word against dictionary 
# check each word against corpus list




### analyse corpus ###
# dictionary of all words/tokens in corpus
  # filter out punctuation groups of 3 or less chars
  # filter out numbers
  # filter out words with non-ascii chars
## zipf plot of corpus
# check all corpus words vs dictionary
# get corpus dictionary match rate
def tokenizeCorpus(dirName):
  '''
  :param dirName: the path to the directory containing the corpus txt file
  :return: tokenized_corpus: a list of lists -> lv1 sentences, lv2 sublists are tokens of that sentence.
            vocabulary: an ordered list of unique tokens in corpus
            vocab_stats: a dictionary storing the frequency of each token (I was curious to see it plotted)
  '''
  print('tokenizing...')
  print(dirName)
  reader = nltk.corpus.reader.PlaintextCorpusReader(dirName, ".*\.txt")
  tokenized_corpus = []   # list of lists. lv1 sentences, lv2 sublists are tokens of that sentence
  vocabulary = []         # an ordered sequence with unique vocabulary items
  vocab_stats_dict = {}   # for plotting to see zipf distribution

  valid_word_cnt = 0
  invalid_word_cnt = 0
  nonAcsiiNumPunc = 0

  for sent_idx, sent in enumerate(reader.sents()):
      sent_lower = []
      for tok in sent:
        if isNotASCII(tok): # avoid non-asciii chars, for ease of plotting and stats
          nonAcsiiNumPunc += 1
          continue
        if re.search(r'\W', tok): # avoid char chunks of punctuation. (tokenizer will have removed legit)
          nonAcsiiNumPunc += 1
          continue
        if re.search(r'[0-9]', tok): #avoid numbers, like dates or years
          nonAcsiiNumPunc += 1
          continue
          
        tok_l = tok.lower()
        sent_lower.append(tok_l)
        if tok_l not in vocabulary:
          vocabulary.append(tok_l)

        if vocab_stats_dict.get(tok_l) == None:
          vocab_stats_dict[tok_l] = 1
        else:
          vocab_stats_dict[tok_l] = vocab_stats_dict.get(tok_l) + 1

        if d.check(tok):
          valid_word_cnt += 1
        else:
          invalid_word_cnt += 1


      tokenized_corpus.append(sent_lower)  # append sentence of tokens
  return tokenized_corpus, vocabulary, vocab_stats_dict, valid_word_cnt, invalid_word_cnt, nonAcsiiNumPunc

def isNotASCII(s):
  try:
    s.encode(encoding='utf-8').decode('ascii')
  except UnicodeDecodeError:
    return True
  else:
    return False



def tokenize_sentence(self, sentence):
  '''
  a helper method, for tokenizing (and lower-casing) string inputs
  :param sentence: 
  :return: 
  '''
  tkns = []
  sent = nltk.word_tokenize(sentence)
  for tok in sent:
    tok_l = tok.lower()
    tkns.append(tok_l)
  return tkns

def plot_zipf(dict):
    tok_keys = sorted(dict, key=dict.__getitem__, reverse=True)
    tok_counts = sorted(dict.values(), reverse=True)

    fig, ax = plt.subplots()
    ax.set_xlabel('word')
    ax.set_ylabel('counts')
    ax.set_title('Curiosity Investigation: Is Corpus Zipfian?')

    plt.bar(tok_keys, tok_counts)
    plt.xticks(rotation=90, fontsize=6)
    plt.show()




corpus, vocab, vocab_stats, valCnt, invalCnt, trash = tokenizeCorpus(file_corpus)
# print(corpus)
# print(vocab)
print(vocab_stats)
print('valid: ',valCnt)
print('invalid: ',invalCnt)
print('trash: ',trash)
print('TOTAL TOKENS: ',trash + valCnt + invalCnt)

print('-------   max item   -------')
# plot_zipf(vocab_stats)
# print(max(vocab_stats, key=vocab_stats.get))
# print(vocab_stats['the'])

# second_max = sorted(set(vocab_stats))[-2]
# print(sorted(set(vocab_stats))[-1])
# print(sorted(set(vocab_stats))[-2])
# print(sorted(set(vocab_stats))[-3])
# print(sorted(set(vocab_stats))[-4])
# print(sorted(set(vocab_stats))[-5])
# print(sorted(set(vocab_stats))[-6])
# print(sorted(set(vocab_stats))[0])
# print(sorted(set(vocab_stats))[1])
# print(sorted(set(vocab_stats))[2])
# print(sorted(set(vocab_stats))[4])
# plot_zipf(vocab_stats)

# print(re.search(r'\W', '!,"'))
# print(re.search(r'\W', ";"))
# print(re.search(r'\W', 'a$'))
# print(re.search(r'\W', '$@^'))
# print(re.search(r'\W', 'av^$!'))
# print(re.search(r'[0-9]', 'abcdefg7'))
