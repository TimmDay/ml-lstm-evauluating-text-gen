from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
import numpy as np
import sys
import random
import matplotlib.pyplot as plt


file = open('corpus_tweets/tweet-corpus_melb_19-4-15.txt', mode='r')
# file = open('corpus_wiki/2019-04-08_wiki_corpus.txt', mode='r')
text = file.read().lower()
print('corpus length: ', len(text))

chars = sorted(list(set(text)))
print('total chars: ', len(text))

char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i,c) for i, c in enumerate(chars))
# print(char_indices)
# print(indices_char)

# cut text into arbitrary sequences of maxlen caracters long
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i: i + maxlen])
  next_chars.append(text[i + maxlen])
print('number of sentences: ', len(sentences))

# Vectorisation
print('vectorising.......')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)#
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# print(len(x))
# print(len(x[0]))
# print(len(x[0][0]))

for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x[i,t, char_indices[char]] = 1
  y[i,char_indices[next_chars[i]]] = 1


# BUILD MODEL - LSTM
print('building lstm')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
optimizer= RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)



def sample(preds, temperature=1.0):
  # function to sample index from probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def on_epoch_end(epoch, _):
  print()
  print('-------- Generating text after epoch num: %d' % epoch)

  start_index = random.randint(0, len(text) - maxlen - 1)
  for diversity in [0.2, 0.5, 1.0, 1.2]:
    print('-------- diversity: ', diversity)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('------- generating with seed: ', sentence)
    text_file= open('output_19-04-15_lstm_char_60_tweet_melb.txt', 'a') #TODO: check output file name
    text_file.write(generated)

    sys.stdout.write(generated)

    for i in range(400):
      x_pred = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.
      
      preds = model.predict(x_pred, verbose=0)[0]
      next_index = sample(preds, diversity)
      next_char = indices_char[next_index]

      generated += next_char
      sentence = sentence[1:] + next_char

      text_file = open('output_19-04-15_lstm_char_60_tweet_melb.txt', 'a') #TODO: check output file name
      text_file.write(next_char)

      sys.stdout.write(next_char)
      sys.stdout.flush()
    print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit(x,y, batch_size=128, epochs=60, callbacks=[print_callback])
model.save('19-04-15_lstm_char_60_tweet_melb.h5') #TODO: check output model name
plt.plot(history.history['loss'])
plt.show()
