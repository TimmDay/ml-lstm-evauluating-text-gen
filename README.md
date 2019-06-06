The code for the training of the Text Generation Model (character-based LSTM) can be found in train_text_gen_char.py

The code for evaluating the corpus and the generated text, and exploring the 'creative validity' of the generated output, can be found in process_gen_text.py.



Twitter Data:
2018-04-04 english : geoloc: London : no capitals, short tweets removed

Wiki Data:
2018-04-08 english : inc capitals & special chars

Models:
twitter 18-04-04 : 70 epochs : LSTM(80) : 0.1 dropout : dense/softmax : compile/catcrossent/adam

wiki    18-04-08 : 70 epochs : LSTM(80) : 0.1 dropout : dense/softmax : compile/catcrossent/adam
