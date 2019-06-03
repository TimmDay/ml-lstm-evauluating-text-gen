import nltk
import re
import enchant
import matplotlib.pyplot as plt

print(enchant.list_languages())

d = enchant.Dict("en_GB") # change to en_US, en_GB, en_AU as required
file_corpus = 'corpus_tweets/london' # or corpus_wiki or corpus_tweets/london
file_gen_text = ''

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


### analyse generated text ###

# go through generated text. 
  # want to be able to select diversity measure (0.2, 0.5, 1.0, 1.2)
  # want to be able to select what epoch to start at (eg 50 - 60)

# remove all seed text from the dictionary being built (it was not generated)

def tokenizeGenerated(dirName):
  '''
  :param dirName: the path to the directory containing the generated txt file
  :return: tokenized_corpus: a list of lists -> lv1 sentences, lv2 sublists are tokens of that sentence.
            vocabulary: an ordered list of unique tokens in corpus
            vocab_stats: a dictionary storing the frequency of each token (I was curious to see it plotted)
  '''
  print('tokenizing generated text...')
  print(dirName)
  reader = nltk.corpus.reader.PlaintextCorpusReader(dirName, ".*\.txt")
  
  #only start recording when certain Epoch is passed
  
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



#ignore lines starting with -, or a number, or 'Epoch'
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


### CORPUS ###

# corpus, vocab, vocab_stats, valCnt, invalCnt, trash = tokenizeCorpus(file_corpus)
# print(corpus)
# print(vocab)
# print(vocab_stats)
# print('valid: ',valCnt)
# print('invalid: ',invalCnt)
# print('trash: ',trash)
# print('TOTAL TOKENS: ',trash + valCnt + invalCnt)

# print('-------   max item   -------')
# plot_zipf(vocab_stats)
# print(max(vocab_stats, key=vocab_stats.get))
# print(vocab_stats['the'])



### GENERATED TEXT ###

def filterForEpochDiversity(filePath, epoch, diversity):
  '''
  :param filePath: the path to the file containing the generated text 
  :param epoch: the epoch at which to start considering output
  :param diversity: the output diversity measure to consider (must be 0.2, 0.5, 1.0 or 1.2)
  :return: an array of strings that pass the filter
  '''
  filtered_text = []
  with open(filePath) as f:
    content = f.readlines()
  
    epoch_target = epoch
    diversity_target = diversity

    epoch_num = 0
    diversity_num = 0
    seed_text = ''

    for line in content:
      # check current Epoch (when seen)
      if line.startswith('Epoch'):
        epoch_num = line[6:8]
        if re.search('/',epoch_num):
          epoch_num = epoch_num[0:1]
        epoch_num = int(epoch_num)

      # considering lines only when in epoch range
      if epoch_num >= epoch_target:
        
        # update current diversity (when seen)
        if line.startswith('-------- diversity:'):
          diversity_num = float(line[-4:])
          continue

        # considering lines within epoch only at relevant diversity
        if diversity_num == diversity_target:

          # store seed text for removal
          if line.startswith('------- generating with seed:'):
            seed_text = line[29:].strip()
            continue
          
          if line.startswith('Epoch'): #we are finished with the text and at next section
            diversity_num = -1

          else:
            if line.find(seed_text) != -1:
              line = line[len(seed_text):] # removes the seed text from the start of the line
            #   print(line)
            # print('EPOCH:', epoch_num, 'DIVERSITY: ', diversity_num)
            # print(line)
            if len(line) > 1:
              filtered_text.append(line)
        else: 
          continue # keep going until within relevant diversity range
      else:
        continue # keep going until passed relevant epoch
  return filtered_text


# 'test/test.txt'
# outputs/twitter_melbourne/2019-06-01_tweet_melb_with_processing.txt
# outputs/twitter_london/2019_tweet_london_with_processing.txt
# outputs/wikipedia/2019_04_04_wiki_with_processing.txt

filtered_output = filterForEpochDiversity('outputs/twitter_melbourne/2019-06-01_tweet_melb_with_processing.txt',50, 1.0)
print(filtered_output)

