# text generation is a type of language modelling problem
# text summarization, conversational system
# learns liklihood of occurence of a word based on previous sequence of words in the text
# LMs can be n-grams, character, sentence level or even <p> level!

# generate NL text
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras.utils as ku
import numpy as np 
import matplotlib.pyplot as plt

# data = """The cat and her kittens
# They put on their mittens,
# To eat a Christmas pie.
# The poor little kittens
# They lost their mittens,
# And then they began to cry.
# O mother dear, we sadly fear
# We cannot go to-day,
# For we have lost our mittens.
# If it be so, ye shall not go,
# For ye are naughty kittens."""

file = open('corpus_wiki/2018-04-08_wiki_corpus.txt', mode='r')
data = file.read()

tokenizer = Tokenizer()
def dataset_prep():
  corpus = data.lower().split("\n")
  # print(corpus)
  tokenizer.fit_on_texts(corpus) # trains the tokenizer model on the corpus. gives us access to:
    # word_counts: A dictionary of words and their counts.
    # word_docs: A dictionary of words and how many documents each appeared in.
    # word_index: A dictionary of words and their uniquely assigned integers.
    # document_count:An integer count of the total number of documents that were used to fit the Tokenizer.
  total_words= len(tokenizer.word_index) + 1

  # print(tokenizer.word_counts) # some words appear more than once
  # print('total_words: ', total_words)
  # print(tokenizer.word_index)

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

# paper points:
# vanishing gradient with large num layers, hence lstm
# dropout layer helps prevent overfitting by randomly turning off some eactivations in he lstm layer
def create_model(predictors, label, max_seq_len, total_words):
  input_len = max_seq_len - 1

  model = Sequential()
  model.add(Embedding(total_words, 10, input_length=input_len))
  # model.add(LSTM((50),return_sequences=True)) # what is the 150 units exactly?
  model.add(LSTM(80)) # what is the 150 units exactly?
  model.add(Dropout(0.1))
  model.add(Dense(total_words, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  history = model.fit(predictors, label, epochs=70, verbose=1)
  plt.plot(history.history['loss'])
  # plt.show()

  model.save('18-04-07_wiki_lstm_80.h5')
  return model


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


# def perplexity(y_true, y_pred):
#     cross_entropy = K.categorical_crossentropy(y_true, y_pred)
#     perplexity = K.pow(2.0, cross_entropy)
#     return perplexity

# EXECUTE
X, Y, max_len, total_words = dataset_prep()
# print(X)
# print(max_len)
# print(total_words)
model = create_model(X,Y,max_len,total_words)

text = generate_text('cat and', 2, max_len, model)
print(text)
text = generate_text('we naughty and', 3, max_len, model)
print(text)
text = generate_text('several big', 2, max_len, model)
print(text)

# a = [0,1,2,3,4,5,6,7,8,9]
# print(a[:-1])
