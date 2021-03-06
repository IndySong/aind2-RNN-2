import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    
    X = [ series[i:i+window_size]  for i in range(0,len(series) -window_size , 1)]
    y = [  series[i+window_size]  for i in range(0,len(series) -window_size , 1)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
	model = Sequential()
	model.add(LSTM(5, input_shape=(window_size, 1)))
	model.add(Dense(1))
	return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
	#we list what we will replace and then replace them
    punctuation = ['!', ',', '.', ':', ';', '?']
    to_replace = [ '\n', '\r' , '\ufeff' , 'à' , 'â','è', 'é' , '@', '*', '&', '#' , '/', '$'  , '-' , '"','+'   , '%', '0', '1','2' , '3' ,'4' ,'5' ,'6' ,'7' , '8','9' , '<' ,'>' , '^', '[', ']' , '{', '}' , '('  ,')' ,  "'", '_'  '`' , '\\', '~' ,'=']
    replace_by = [ ' ',  ' ' , ' '       , 'a' , 'a','e', 'e' , ' ', ' ', ' ', ' ' , ' ', ' '  , ' ' , ' ', ' '  , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ', ' ' , ' ', ' ', ' ', ' ' , ' ', ' ', ' ' , ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    alphabet = [ '!', ',', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    #replace the ones we know
    for i in range(len(to_replace)):
    	text= text.replace(to_replace[i],replace_by[i])
    chars = sorted(list(set(text)))
    #for the udacity test, there are some remaining
    for a in chars:
    	if(a not in  alphabet):
    		text= text.replace(a,' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [ text[i:(i+window_size)]   for i in range(0,len(text) -window_size , step_size)]
    outputs = [text[i+window_size]  for i in range(0,len(text) -window_size , step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
	#dropout added
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    #model.add(Dense(num_chars))
    model.add(Dense(num_chars, activation='softmax'))
    return model
