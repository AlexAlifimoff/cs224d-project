import keras as K
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import *
import numpy as np

def generate_input_pairs_for_data(x, y, context_size, eos_token):
    batch_sz = x.shape[0]
    assert(batch_sz == y.shape[0])
    output_x = []
    output_y = []
    labels = []
    for i in range(batch_sz):
        cur_x = x[i, :]
        cur_y = y[i, :]
        y_windows = []
        for offset in range(y.shape[1]-context_size):
            window = cur_y[offset:offset + context_size]
            y_windows.append(window)
            to_predict = cur_y[offset+context_size]
            labels.append(to_predict)
            if to_predict == eos_token: break
        x_copies = [cur_x] * len(y_windows)
        y_windows = np.array(y_windows)
        output_x.append(x_copies)
        output_y.append(y_windows)      
    return np.vstack(output_x), np.vstack(output_y), np.array(labels)
        
            

class LogRegBaseline(object):
    def __init__(self, context_size, embedding_size, embedding_matrix, input_length,
                    vocab_sz, summary_length, batch_sz, num_batches = 10, eos_token = None):
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.input_length = input_length
        self.vocab_sz = vocab_sz
        self.summary_length = summary_length
        self.num_batches = num_batches
        self.eos_token = eos_token
        self.hidden_sz = 100

        self.initialize(batch_sz)


        self.params = []

    def initialize(self, batch_sz):
        y_embedding = Sequential()
        y_embedding.add(Embedding(self.vocab_sz, self.embedding_size, weights = [self.embedding_matrix.T], input_length=self.context_size)) 
        y_embedding.add(Flatten())
        
        x_embedding = Sequential()
        x_embedding.add(Embedding(self.vocab_sz, self.embedding_size, weights = [self.embedding_matrix.T], input_length=self.input_length)) 
        x_embedding.add(Flatten())

        model = Sequential()
        model.add(Merge([y_embedding, x_embedding], mode='concat', concat_axis=1))
        model.add(Dense(self.hidden_sz,
            input_dim = (self.summary_length + self.input_length) * self.embedding_size,
            activation = 'tanh'))

        model.add(Dense(self.vocab_sz, input_dim = self.hidden_sz, activation = 'softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model = model

        self.f_conditional_probability_distribution = self.conditional_probability_distribution 

    def conditional_probability_distribution(self, inpt, summ, idx):
        idx = idx - self.context_size
        x, y, labels = generate_input_pairs_for_data(inpt, summ, self.context_size, self.eos_token)
        y = np.array([y[idx].T])
        x = np.array([x[idx].T])
        return self.model.predict_on_batch([y, x]) 

    def train_one_superbatch(self, inputs, summaries, imasks, smasks, lr, batches_per_update):
        x, y, labels = generate_input_pairs_for_data(inputs, summaries, self.context_size, self.eos_token)
        cost = self.model.train_on_batch([y, x], labels)
        print("batch cost: ", cost)

    def validate(self, inputs, summaries):
        x, y, labels = generate_input_pairs_for_data(inputs, summaries, self.context_size, self.eos_token)
        cost = self.model.test_on_batch([y, x], labels)

        print("validate cost: ", cost)
