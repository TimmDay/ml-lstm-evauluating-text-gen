from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ku

# file = open('corpus_wiki/2018-04-08_wiki_corpus.txt', mode='r')
file = open('corpus_tweets/2018-04-04_tweet-corpus.txt', mode='r')
data = file.read()
# model = load_model('saved_models/18-04-07_wiki_lstm_80.h5')
model = load_model('saved_models/18-04-04_tweet_lstm_80.h5')
tokenizer = Tokenizer()

def dataset_prep():
  corpus = data.lower().split("\n")
  # print(corpus)
  tokenizer.fit_on_texts(corpus) # trains the tokenizer model on the corpus. gives us access to:
    # word_counts: A dictionary of words and their counts.
    # word_docs: A dictionary of words and how many documents each appeared in.
    # word_index: A dictionary of words and their uniquely assigned integers.
    # document_count:An integer count of the total number of documents that were used to fit the Tokenizer.
  total_words = len(tokenizer.word_index) + 1

  # convert corpus to a flat dataset of sentence sequences
  input_sequences= []
  for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0] # returns vectors with token indices instead of words
    for i in range(1,len(token_list)):
      n_gram_sequence = token_list[:i+1] # slice from 0 to i+1. first iter is 0 to 2 (2 els)
      input_sequences.append(n_gram_sequence)

  # pad sequences with zeroes so all are the same length
  max_seq_len = max([len(x) for x in input_sequences]) # give arr of lengths, reduce to the max
  input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len,padding='pre'))

  # we need predictors and label data (input and gold standard),
  # use last word of ngram as label, and the sequence before as predictor
  """
  Sentence: "they are learning data science"
  PREDICTORS             | LABEL
  they                   | are
  they are               | learning
  they are learning      | data
  they are learning data | science
  """
  predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
  label = ku.to_categorical(label,num_classes=total_words) # will get one hots
  return predictors, label, max_seq_len, total_words




# input words are the seed text
def generate_text(seed_text, next_words, max_seq_len, model):
  for j in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)

    output_word = ''
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += ' ' + output_word
  return seed_text


X, Y, max_len, total_words = dataset_prep()

text = generate_text('the big', 8, max_len, model)
print(text)
text = generate_text('we naughty and', 8, max_len, model)
print(text)
text = generate_text('several big', 2, max_len, model)
print(text)
text = generate_text('', 20, max_len, model)
print(text)
text = generate_text('', 25, max_len, model)
print(text)
text = generate_text('', 28, max_len, model)
print(text)
text = generate_text('a', 25, max_len, model)
print(text)
text = generate_text('the', 18, max_len, model)
print(text)
text = generate_text('one', 15, max_len, model)
print(text)

text = generate_text('two', 9, max_len, model)
print(text)
text = generate_text('small', 9, max_len, model)
print(text)
text = generate_text('happy', 15, max_len, model)
print(text)
text = generate_text('outrageous', 12, max_len, model)
print(text)

text = generate_text('three', 15, max_len, model)
print(text)
text = generate_text('four', 15, max_len, model)
print(text)
text = generate_text('for', 15, max_len, model)
print(text)
text = generate_text('once', 15, max_len, model)
print(text)

text = generate_text('several', 15, max_len, model)
print(text)
text = generate_text('a large kitten', 15, max_len, model)
print(text)
text = generate_text('dinner', 15, max_len, model)
print(text)
text = generate_text('football', 15, max_len, model)
print(text)
text = generate_text('joy', 15, max_len, model)
print(text)

text = generate_text('happy', 15, max_len, model)
print(text)
text = generate_text('love', 15, max_len, model)
print(text)

text = generate_text('love is', 15, max_len, model)
print(text)

text = generate_text('I love', 15, max_len, model)
print(text)
text = generate_text('so happy', 15, max_len, model)
print(text)