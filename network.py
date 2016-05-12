import theano
import theano.tensor as T
import numpy as np
import optimisers
from six.moves import cPickle as pickle
import os

import sys
sys.setrecursionlimit(5000)

class EmbeddingsLayer(object):
    #
    # input_matrix is v-by-c array, with v being size of vocab and c size of context
    #
    def __init__(self, input_vec, context_size, embedding_size, vocabulary_size, flatten = True, emb_name = 'E', embedding_matrix = None):
        if embedding_matrix is None:
            value = np.random.uniform(low=-0.02, high=0.02, size=(embedding_size, vocabulary_size))
        else: value = embedding_matrix
        self.E = theano.shared(
                value = value.astype(theano.config.floatX),
                name = emb_name,
                borrow = True
                )

        self.input = input_vec 

        self.params = [self.E]

        self._output_function = self.E[:,input_vec]
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
    def __init__(self, x, y, embedding_size, num_input_words, vocab_sz, y_context_sz, embedding_matrix = None):
        print("building embeddings layer with expected output...", embedding_size, "by", num_input_words)
        self.EmbeddingsLayer = EmbeddingsLayer(x, num_input_words, embedding_size,
                vocab_sz, flatten = False, emb_name = 'F', embedding_matrix = embedding_matrix)

        p = np.divide(np.ones(num_input_words), num_input_words)
        print("p length", num_input_words)
        print("embedding_size", embedding_size)
        p = T.as_tensor_variable(p) 
        self.input = x
        self.params = self.EmbeddingsLayer.params

        x_tilde = self.EmbeddingsLayer.output 
        self.output = T.dot(p, x_tilde.T).reshape( (embedding_size, 1) )

    def func(self, x, y):
        return theano.function( [x, y], self.output, on_unused_input='ignore' )

class AttentionEncoder(object):
    def __init__(self, x, y, input_embedding_size, summary_embedding_size, num_input_words, vocab_sz, y_context_sz):
        self.InputEmbeddingsLayer = EmbeddingsLayer(x, num_input_words, input_embedding_size, vocab_sz,
                flatten = False, emb_name = 'F')

        x_tilde = self.InputEmbeddingsLayer.output

        self.SummaryEmbeddingsLayer = EmbeddingsLayer(y, y_context_sz, summary_embedding_size, vocab_sz, flatten = True,
                emb_name = 'G')

        P_value = np.random.uniform(low=-0.02, high=0.02, size=(input_embedding_size, y_context_sz * summary_embedding_size))
        self.P = theano.shared(
                    value = P_value.astype(theano.config.floatX),
                    name = 'P', 
                    borrow = True)

        Q = 2

        m = self.build_attention_convolution_matrix(Q, num_input_words)

        y_tilde = self.SummaryEmbeddingsLayer.output
        p =  T.dot(x_tilde.T, T.dot(self.P, y_tilde)).T  
        p = T.nnet.softmax( p ).T


        m_dot_p = T.dot(m, p)
        self.output = T.dot(x_tilde, m_dot_p)

        self.params = [self.P] + self.InputEmbeddingsLayer.params + self.SummaryEmbeddingsLayer.params

    def build_attention_convolution_matrix(self, Q, l):
        # straight up borrowed from Jencir's initial model
        assert l >= Q
        m = np.diagflat([1.] * l)
        for i in range(1, Q):
            m += np.diagflat([1.] * (l - i), k=i)
            m += np.diagflat([1.] * (l - i), k=-i)
        return m / np.sum(m, axis = 0)


class OutputLayer(object):
    def __init__(self, encoder, h, vocab_sz, hidden_layer_sz, embedding_size):
        self.input = [encoder, h]
        V_value = np.random.uniform(low=-0.02, high=0.02, size=(vocab_sz, hidden_layer_sz))
        self.V = theano.shared(
                    value = V_value.astype(theano.config.floatX),
                    name = 'V', 
                    borrow = True)

        W_value = np.random.uniform(low=-0.02, high=0.02, size=(vocab_sz, embedding_size))
        self.W = theano.shared(
                    value = W_value.astype(theano.config.floatX),
                    name = 'W',
                    borrow = True)

        encoder_output = T.dot(self.W, encoder.output)
        language_model_output = T.dot(self.V, h.output)

        self.output = encoder_output + language_model_output #T.dot(self.V, h.output) + T.dot(self.W, encoder.output)
        #self.output = theano.printing.Print("output layer pre softmax")(self.output)
        self.output = T.nnet.softmax(self.output.T).T
        #self.output = theano.printing.Print("output layer post softmax")(self.output)

        self.params = encoder.params + h.params + [self.V, self.W]

    def func(self, x, y):
        return theano.function( [x, y], self.output )

class SummarizationNetwork(object):
    def __init__(self, input_sentence_length = 5, vocab_size = 4, embedding_size = 20,
            context_size = 3, hidden_layer_size = 10, embedding_matrix = None, l2_coefficient=0.05,
            encoder_type = 'attention', batch_size = 10, summary_length = 10):
        assert(encoder_type in ['attention', 'bow'])
        self.encoder_type = encoder_type
        self.input_sentence_length = input_sentence_length 
        self.vocab_size = vocab_size 
        self.embedding_size = embedding_size 
        self.context_size = context_size 
        self.hidden_layer_size = hidden_layer_size 

        self.embedding_matrix = embedding_matrix
        self.l2_coefficient = l2_coefficient

        self.summary_length = 8#summary_length
        self.batch_size = 10#batch_size

        self.initialize()

    def conditional_probability_distribution(self, x, y):
        el = EmbeddingsLayer(y, self.context_size, self.embedding_size,
                self.vocab_size, flatten = True, embedding_matrix = self.embedding_matrix)
        hl_1 = HiddenLayer(el, self.embedding_size * self.context_size,
                self.hidden_layer_size)
        if self.encoder_type == 'bow':
            enc = BagOfWordsEncoder(x, y, self.embedding_size, self.input_sentence_length,
                    self.vocab_size, self.context_size, embedding_matrix = self.embedding_matrix)
            self.embedding_matrices = [enc.EmbeddingsLayer.params[0], el.params[0]]
        elif self.encoder_type == 'attention':
            enc = AttentionEncoder(x, y, self.embedding_size, self.embedding_size,
                    self.input_sentence_length, self.vocab_size, self.context_size)

            self.embedding_matrices = [enc.InputEmbeddingsLayer.params[0],
                    enc.SummaryEmbeddingsLayer.params[0], el.params[0]]

        output_layer = OutputLayer(enc, hl_1, self.vocab_size, self.hidden_layer_size,
                self.embedding_size)
        self.layers = [el, hl_1, enc, output_layer]
        self.params = output_layer.params
        self._conditional_probability_distribution = output_layer.output
        self.get_conditional_probability_distribution = theano.function([x, y], output_layer.output)#, allow_input_downcast=True)
        return output_layer.output         

    def conditional_probability(self, x, y, y_position):
                
        dist = self.conditional_probability_distribution(x,
                y[y_position-self.context_size:y_position])

        self.f_conditional_probability_distribution = theano.function([x, y, y_position], dist)
        self.get_conditional_probability_distribution_for_token = self.f_conditional_probability_distribution
        return dist[y[y_position]]

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
        func = lambda i, documents, summaries: T.sum(self.negative_log_likelihood(documents[i, :],
            summaries[i, :]), axis=1)

        probs, _ = theano.scan(func, sequences=[document_range], non_sequences=[documents,summaries])
        return probs.sum() 

    def train_model_func(self, batch_size):
        docs = T.imatrix('docs')
        summaries = T.imatrix('summaries')

        cost = self.negative_log_likelihood_batch(docs, summaries, batch_size)
        regularization_cost = self.l2_coefficient * sum([(p ** 2).sum() for p in self.params])

        self.get_batch_cost_unregularized = theano.function([docs, summaries], cost, allow_input_downcast=True)
        theano.printing.debugprint(cost)
        cost = cost + regularization_cost

        #cost = theano.printing.Print("cost")(cost)
        #theano.pp(cost)
        #self.params = [theano.printing.Print("...")(p) for p in self.params]
        params = {p.name: p for p in self.params} 
        #print(list(params.values()))
        #print(self.params)
        grads = T.grad(cost, self.params)
        #grads = theano.printing.Print("grads")(grads)

        # learning rate
        lr = T.scalar(name='lr')
        gradient_update, network_update = optimisers.sgd(lr, self.params,
                                                                grads, docs, summaries, cost)  
        return gradient_update, network_update

    def normalize_embeddings_func(self, mode = "matrix"):
        embeddings = self.embedding_matrices
        updates = []
        for matrix in embeddings:
            if mode == "vector":
                norm_matrix = matrix / matrix.sum(axis = 0).reshape((1, matrix.shape[1]))
            elif mode == "matrix":
                norm_matrix = matrix / T.max(matrix.sum(axis = 0))
            updates.append( (matrix, norm_matrix) )

        return theano.function([], embeddings, updates=updates)

    def initialize(self):
        self.gradient_update, self.network_update = self.train_model_func(self.batch_size)

    def train_one_batch(self, documents, summaries, learning_rate, verbose = False):
        cost = self.gradient_update(documents, summaries)
        self.network_update(learning_rate)

        if verbose:
            print("Cost after update: ", cost)

        return cost

    def save(self, name):
        f = open(name, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, name):
        if os.path.isfile(name):
            f = open(name, 'rb')
            s = pickle.load(f)
            f.close()
            print("network loaded succesfully")
            return s
        else:
            print("tried to load network -- no file found")

