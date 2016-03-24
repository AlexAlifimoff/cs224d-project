import theano
import theano.tensor as T
import numpy as np

class EmbeddingsLayer(object):
    #
    # input_matrix is v-by-c array, with v being size of vocab and c size of context
    #
    def __init__(self, input_matrix, context_size, embedding_size, vocabulary_size, flatten = True, emb_name = 'E'):
        self.E = theano.shared(
                value = np.zeros((embedding_size, vocabulary_size),
                        dtype = theano.config.floatX),
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
        self.U = theano.shared(
                value = np.zeros( (output_size, input_size),
                        dtype = theano.config.floatX),
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
        self.EmbeddingsLayer = EmbeddingsLayer(x, num_input_words, embedding_size, vocab_sz, flatten = False, emb_name = 'F')

        p = np.divide(np.ones(num_input_words), num_input_words)
        #p.reshape( (
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
        self.V = theano.shared(
                    value = np.zeros( (vocab_sz, hidden_layer_sz),
                        dtype = theano.config.floatX),
                    name = 'V', 
                    borrow = True)
        self.W = theano.shared(
                    value = np.zeros( (vocab_sz, hidden_layer_sz),
                        dtype = theano.config.floatX),
                    name = 'W',
                    borrow = True)

        self.output = T.dot(self.V, h.output) + T.dot(self.W, encoder.output)
        self.output = T.exp(self.output)

        self.params = encoder.params + h.params + [self.V, self.W]

    def func(self, x, y):
        return theano.function( [x, y], self.output )

class SummarizationNetwork(object):
    def __init__(self):
        

        self.input_sentence_length = 5

        self.vocab_size = 4 
        self.embedding_size = 2 
        self.context_size = 2
        self.hidden_layer_size = 10

        y_input = np.matrix("0, 0 ; \
                             1, 0 ; \
                             0, 0 ; \
                             0, 1 " )

        x_input = np.matrix("  0, 0, 0, 0, 1 ; \
                               1, 0, 0, 0, 0 ; \
                               0, 0, 0, 1, 0 ; \
                               0, 1, 1, 0, 0  " )

        self.y = T.matrix('y')
        self.x = T.matrix('x')

        el = EmbeddingsLayer(self.y, self.context_size, self.embedding_size,
                self.vocab_size, flatten = True)
        hl_1 = HiddenLayer(el, self.embedding_size * self.context_size,
                self.hidden_layer_size)
        enc_bow = BagOfWordsEncoder(self.x, self.y, self.hidden_layer_size,
                self.input_sentence_length, self.vocab_size, self.context_size)

        self.output_layer = OutputLayer(enc_bow, hl_1, self.vocab_size, self.hidden_layer_size)
        
        self.layers = [el, hl_1, enc_bow, self.output_layer]
        self.conditional_probability = self.output_layer.func(self.x, self.y)
        print(self.conditional_probability(x_input, y_input))
        self.params = self.output_layer.params

    def negative_log_likelihood(x, y):
        pass




s = SummarizationNetwork()
print(s.params)
