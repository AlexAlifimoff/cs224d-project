import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import optimisers
from six.moves import cPickle as pickle
import os

class EmbeddingsLayer(object):
    #
    # input_matrix is v-by-c array, with v being size of vocab and c size of context
    #
    def __init__(self, input_matrix, context_size, embedding_size, vocabulary_size, flatten = True, emb_name = 'E'):
        value = np.random.uniform(low=-0.02, high=0.02, size=(embedding_size, vocabulary_size))
        self.E = theano.shared(
                value = value.astype(theano.config.floatX),
                name = emb_name,
                borrow = True
                )

        self.input = input_matrix 

        self.params = [self.E]

        self._output_function = T.dot(self.E, input_matrix)
        if flatten:
            self.output = T.reshape(self._output_function, (embedding_size * context_size, 1))
        else:
            self.output = self._output_function

    def func(self, inpt):
        return theano.function( [inpt], self.output )

class HiddenLayer(object):
    def __init__(self, input, input_size, output_size ):
        value = np.random.uniform(low=-0.02, high=0.02, size=(output_size, input_size))
        self.U = theano.shared(
                value = value.astype(theano.config.floatX),
                name = 'U',
                borrow = True
                )
        self.input = input.output
        self.params = [self.U] + input.params

        self.output = T.tanh( T.dot(self.U, self.input) )

    def func(self, inpt):
        return theano.function( [inpt], self.output )

class BagOfWordsEncoder(object):
    def __init__(self, x, y, embedding_size, num_input_words, vocab_sz, y_context_sz):
        self.EmbeddingsLayer = EmbeddingsLayer(x, num_input_words, embedding_size,
                vocab_sz, flatten = False, emb_name = 'F')

        p = np.divide(np.ones(num_input_words), num_input_words)
        print("p length", num_input_words)
        p = T.as_tensor_variable(p) 
        self.input = x
        self.params = self.EmbeddingsLayer.params

        x_tilde = self.EmbeddingsLayer.output 
        self.output = T.dot(p, x_tilde.T).reshape( (embedding_size, 1) )

    def func(self, x, y):
        return theano.function( [x, y], self.output, on_unused_input='ignore' )

class OutputLayer(object):
    def __init__(self, encoder, h, vocab_sz, hidden_layer_sz):
        self.input = [encoder, h]
        V_value = np.random.uniform(low=-0.02, high=0.02, size=(vocab_sz, hidden_layer_sz))
        self.V = theano.shared(
                    value = V_value.astype(theano.config.floatX),
                    name = 'V', 
                    borrow = True)

        W_value = np.random.uniform(low=-0.02, high=0.02, size=(vocab_sz, hidden_layer_sz))
        self.W = theano.shared(
                    value = W_value.astype(theano.config.floatX),
                    name = 'W',
                    borrow = True)

        self.output = T.dot(self.V, h.output) + T.dot(self.W, encoder.output)
        self.output = T.exp(self.output)

        self.params = encoder.params + h.params + [self.V, self.W]

    def func(self, x, y):
        return theano.function( [x, y], self.output )

class SummarizationNetwork(object):
    def __init__(self, input_sentence_length = 5, vocab_size = 4, embedding_size = 2,
            context_size = 2, hidden_layer_size = 10):
        self.input_sentence_length = input_sentence_length 
        self.vocab_size = vocab_size 
        self.embedding_size = embedding_size 
        self.context_size = context_size 
        self.hidden_layer_size = hidden_layer_size 
        self.learning_rate = 0.001

        print("input sentence length", self.input_sentence_length)
        print("vocab size", self.vocab_size)

        self.summary_length = 8 
        self.batch_size = 3 

    def conditional_probability_distribution(self, x, y):
        el = EmbeddingsLayer(y, self.context_size, self.embedding_size,
                self.vocab_size, flatten = True)
        hl_1 = HiddenLayer(el, self.embedding_size * self.context_size,
                self.hidden_layer_size)
        enc_bow = BagOfWordsEncoder(x, y, self.hidden_layer_size,
                self.input_sentence_length, self.vocab_size, self.context_size)
        output_layer = OutputLayer(enc_bow, hl_1, self.vocab_size, self.hidden_layer_size)
        self.layers = [el, hl_1, enc_bow, output_layer]
        self.params = output_layer.params
        self._conditional_probability_distribution = output_layer.output
        self.f_conditional_probability_distribution = theano.function([x, y], output_layer.output)
        return output_layer.output
        
    def conditional_probability(self, x, y, y_position):
                
        dist = self.conditional_probability_distribution(x,
                y[:, y_position-self.context_size:y_position])


        self.f_conditional_probability_distribution = theano.function([x, y, y_position], dist)

        return T.dot(dist.T, y[:, y_position])

    def negative_log_likelihood(self, x, y):
        # Here, y is an entire summary and x is an entire text.
        # We start at the first y, take its context and it, and then
        # calculate the probability of our current word appearing
        
        # Since we pad the input vectors, we start at C and go to len(y_original) + C
        y_position_range = T.arange(self.context_size, self.summary_length + self.context_size)

        # We start by indexing into y
        # current_word = y[:, i]
        # context = y[:, i-1-C:i-1]

        func = lambda y_pos, x, y: self.conditional_probability(x, y, y_pos)

        probabilities, _ = theano.scan(func, sequences=y_position_range,
                                    non_sequences = [x, y], n_steps=self.summary_length)

        return -T.log(probabilities.T)

    def negative_log_likelihood_batch(self, documents, summaries, batch_size):
        document_range = T.arange(0, batch_size)
        func = lambda i, documents, summaries: T.sum(self.negative_log_likelihood(documents[i,:,:],
                                                                    summaries[i, :, :]), axis=1)
        func2 = lambda document, summary: T.sum(self.negative_log_likelihood(document,
                                                                    summary), axis=1)
 

        probs, _ = theano.scan(func2, sequences=[documents, summaries],
                                  non_sequences=[])
        return probs.sum() 

    def train_model_func(self, batch_size):
        docs = T.tensor3('docs')
        summaries = T.tensor3('summaries')
        cost = self.negative_log_likelihood_batch(docs, summaries, batch_size)
        params = {p.name: p for p in self.params} 
        grads = T.grad(cost, list(params.values()))

        # learning rate
        lr = T.scalar(name='lr')
        gradient_update, update = optimisers.sgd(lr, params, grads, docs, summaries, cost)
        return gradient_update, update

    def initialize(self):
        grad_update, update = self.train_model_func(self.batch_size)
        self.gradient_update = grad_update
        self.cost_update = update
        return grad_update, update

    def save(self, name):
        f = open(name, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def train_one_epoch(self, documents, summaries):
        num_data_points = documents.shape[0]
        assert(num_data_points == summaries.shape[0])
        num_batches = num_data_points / self.batch_size
        for i in range(num_batches):
            end = (i+1) * self.batch_size
            if end > num_data_points: end = num_data_points
            cur_documents = documents[i*self.batch_size:end, :, :]

    def load(self, name):
        if os.path.isfile(name):
            f = open(name, 'rb')
            s = pickle.load(f)
            f.close()
            print("network loaded succesfully")
            return s
        else:
            print("tried to load network -- no file found")

