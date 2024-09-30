import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.api.preprocessing.sequence import pad_sequences
from keras.src.utils import to_categorical
from keras.src.models import Sequential
from keras.src.layers import Embedding,Dense,Dropout,Bidirectional,LSTM
from keras.src.regularizers import regularizers



data = open('data.txt').read()
corpus = data.lower().split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_word = len(tokenizer.word_index) + 1
# print(corpus)

#create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        # print(n_gram_sequence)
        input_sequences.append(n_gram_sequence)

#pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

#create predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = to_categorical(label, num_classes=total_word)

#build model
model = Sequential()
model.add(Embedding(input_dim=total_word, output_dim=100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(total_word//2, activation='relu', kernel_regularizer=regularizers.L2(0.01)))
model.add(Dense(total_word, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(predictors, label, epochs=100, verbose=1)
model.save('word_predictor.keras')


#predict word
test_seq = 'despite of wrinkles'
next_word = 10

for _ in range(next_word):
    token_list = tokenizer.texts_to_sequences([test_seq])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list)
    predicted_classes = np.argmax(predicted, axis=1)
    output_word = ""
    for word,index in tokenizer.word_index.items():
        if index == predicted_classes:
            output_word = word
            break
    test_seq += " " + output_word
print(test_seq)
