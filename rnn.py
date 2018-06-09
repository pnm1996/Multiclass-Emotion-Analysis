import numpy as np 
import pandas as pd
from sklearn.externals import joblib
from multiprocessing import Queue
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

data = pd.read_csv('book3.csv')
data = data[['text','sentiment']]
data.sentiment = data.sentiment.astype(str)
data.text = data.text.astype(str)		
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
embed_dim = 128
lstm_out = 196

#embed_dim = 200
#lstm_out = 350


model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 32
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
#print("score: %.2f" % (score))
print("accuracy: %.2f" % (acc))

twt = "fucking sad"

twt = tokenizer.texts_to_sequences(twt)

twt = pad_sequences(twt, maxlen=32, dtype='int32', padding='post', truncating='post', value=0)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
print(sentiment)
print(np.argmax(sentiment))
if(np.argmax(sentiment) == 0):
    print("anger")
elif (np.argmax(sentiment) == 1):
    print("fear")
elif (np.argmax(sentiment) == 2):
    print("joy")
