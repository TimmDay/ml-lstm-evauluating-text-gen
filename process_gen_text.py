import nltk
import re
import enchant
import matplotlib.pyplot as plt

print(enchant.list_languages())

# 'corpus_tweets/melbourne'
# 'outputs/twitter_melbourne/2019-06-01_tweet_melb_with_processing.txt'
# 'corpus_tweets/london'
# 'outputs/twitter_london/2019_tweet_london_with_processing.txt'
# 'corpus_wiki'
# 'outputs/wikipedia/2019_04_04_wiki_with_processing.txt'

file_corpus = 'corpus_tweets/melbourne'
file_gen_text = 'outputs/twitter_melbourne/2019-06-01_tweet_melb_with_processing.txt'
d = enchant.Dict("en_AU") # change to en_US, en_GB, en_AU as required
diversity = 1.2 # diversity of generated text to filter for (0.2, 0.5, 1.0 or 1.2)
epoch = 50      # epoch of generated text to start gethering from

#  Construct the dictionary of all vocab in the corpus
def evaluateCorpus(dirName):
  '''
  :param 
    dirName: the path to the directory containing the corpus txt file
  :return 
    vocab_dict: a dictionary storing the frequency of each token (I was curious to see it plotted)
    tokenized_corpus: a list of lists -> lv1 sentences, lv2 sublists are tokens of that sentence.
    vocabulary: an ordered list of unique tokens in corpus
  '''
  print('evaluating corpus...')
  print(dirName)
  reader = nltk.corpus.reader.PlaintextCorpusReader(dirName, ".*\.txt")
  vocab_dict = {}
  tokenized_corpus = []
  vocabulary = []
  
  valid_word_cnt = 0
  invalid_word_cnt = 0
  nonAcsiiNumPunc = 0

  for idx, sent in enumerate(reader.sents()):
      sent_lower = []

      for tok in sent:
        if isNotASCII(tok): # avoid tokens with non-asciii chars. cheap way to filter for english
          nonAcsiiNumPunc += 1
          continue
        if re.search(r'\W', tok): # avoid chunks of punctuation
          nonAcsiiNumPunc += 1
          continue
        if re.search(r'[0-9]', tok): # avoid numbers, like dates or years
          nonAcsiiNumPunc += 1
          continue
          
        tok_l = tok.lower()
        sent_lower.append(tok_l)
        if tok_l not in vocabulary:
          vocabulary.append(tok_l)

        if vocab_dict.get(tok_l) == None:
          vocab_dict[tok_l] = 1
        else:
          vocab_dict[tok_l] = vocab_dict.get(tok_l) + 1

        if d.check(tok): # dictionary check
          valid_word_cnt += 1
        else:
          invalid_word_cnt += 1
      tokenized_corpus.append(sent_lower)  # append sentence of tokens
  
  # print('valid: ',valCnt)
  # print('invalid: ',invalCnt)
  # print('trash: ',trash)
  # print('TOTAL TOKENS: ',trash + valCnt + invalCnt)
  return vocab_dict, tokenized_corpus, vocabulary

# helper function to assess if a character is not ASCII
def isNotASCII(s):
  try:
    s.encode(encoding='utf-8').decode('ascii')
  except UnicodeDecodeError:
    return True
  else:
    return False

# helper function to tokenise a sentence using nltk word_tokenize
def tokenize_sentence(sentence):
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

# utility function, for plotting a ranked curve of word frequencies in the dictionary, in order to observe zipfian behaviour
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

# Filter the generated text, choosing which epoch to start considering it, and which diversity measure (0.2, 0.5, 1.0 or 1.2) to use
def filterForEpochDiversity(filePath, epoch_target, diversity_target):
  '''
  :param:
    filePath: the path to the file containing the generated text 
    epoch: the epoch at which to start considering output
    diversity: the output diversity measure to consider (must be 0.2, 0.5, 1.0 or 1.2)
  :return:
    filtered_text: an array of strings that pass the filter
  '''
  filtered_text = []
  with open(filePath) as f:
    content = f.readlines()

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

          # Do NOT include seed text in generated corpus
          # so store seed text for removal later. 
          if line.startswith('------- generating with seed:'):
            seed_text = line[29:].strip()
            continue
          
          if line.startswith('Epoch'): # we are finished with the text and at next section
            diversity_num = -1

          else:
            if line.find(seed_text) != -1:
              line = line[len(seed_text):] # removes the seed text from the start of the line

            if len(line) > 1: # filter out random punctuation and single chars
              filtered_text.append(line)
        else: 
          continue # keep going until within relevant diversity range
      else:
        continue # keep going until passed relevant epoch
  return filtered_text

# evalute the (filtered) generated text, counting tokens that:
# - pass an english dictionary test
# - match a token in the corpus dictionary
# - pass the english dictionary but DO NOT match the corpus dictionary (ie: creatively valid)
def evaluateFilteredGeneratedText(arr, vocab_dict):
  '''
  :param:
    arr: an array containing the valid tokens that were filtered from the generated text file
    vocab_dict: .a dictionary of all tokens present in the corpus
  :return:
    creatively_valid: an array of the tokens that are valid in the english dictionary 
      but do NOT appear in the corpus dictionary
  '''
  total_tok = 0
  dict_pass = 0
  corpus_pass = 0
  dict_not_corpus = 0
  neither = 0
  creatively_valid = []

  for sen in arr:
    tokens = tokenize_sentence(sen)
    # print(tokens)
    for tok in tokens:
      tok = tok.lower()
      # is it punctuation? (dont count)
      if re.search(r'\W', tok):
        continue
      # is it a number, dont count
      if re.search(r'[0-9]', tok): 
        continue
      
      total_tok += 1
      if d.check(tok): # dictionary check
        dict_pass += 1
      if tok in vocab_dict: # corpus check
        corpus_pass += 1
      if d.check(tok) and tok not in vocab_dict: # creatively valid check
        dict_not_corpus += 1
        # have we seen this yet?
        if tok not in creatively_valid:
          creatively_valid.append(tok)
        
      if not d.check(tok) and tok not in vocab_dict:
        neither += 1

  print('DICT PASS: ', dict_pass)
  print('CORPUS PASS: ', corpus_pass)
  print('DICT NOT CORPUS: ', dict_not_corpus)
  print('NEITHER: ', neither)
  print('TOTAL TOKENS: ', total_tok)
  return creatively_valid


### MAIN ###
vocab_dict, corpus, vocab = evaluateCorpus(file_corpus)
filtered_gen_output = filterForEpochDiversity(file_gen_text, epoch, diversity)
creatively_valid = evaluateFilteredGeneratedText(filtered_gen_output, vocab_dict)
print(creatively_valid)

# print(corpus)
# print(vocab)
# print(vocab_dict)
# plot_zipf(vocab_dict)