import optimisers
import theano.tensor as T
import theano
import numpy as np
import network

class LogRegBaseline(object):
    def __init__(self, context_size, embedding_size, embedding_matrix, input_length,
                    vocab_sz, summary_length, batch_sz):
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.input_length = input_length
        self.vocab_sz = vocab_sz
        self.summary_length = summary_length

        self.initialize(batch_sz)

    def conditional_probability_distribution(self, x, y):

        self.context_embedding = network.EmbeddingsLayer(x, self.context_size, self.embedding_size,
                                    self.vocab_size, True, 'E', self.embedding_matrix)
        self.input_embedding = network.EmbeddingsLayer(y, self.input_length, self.embedding_size,
                                    self.vocab_size, True, 'F', self.embedding_matrix)

        self.word_vecs = T.concatenate(self.context_embedding.output, self.input_embedding.output)
        # size should be context_size * embedding_size + input_length * embedding_size

        input_sz = (self.context_size + self.input_length) * self.embedding_size

        V_value = np.random.uniform(low=-0.02, high=0.02, size=(self.vocab_sz, input_sz))
        self.V = theano.shared(
                    value = V_value.astype(theano.config.floatX),
                    name = 'V', 
                    borrow = True)

        self.params = [self.V] + self.context_embedding.params + self.input_embedding.params

        return T.dot(self.V, self.word_vecs)


    def conditional_probability(self, x, y, y_position):
                
        dist = self.conditional_probability_distribution(x,
                y[y_position-self.context_size:y_position])

        self.f_conditional_probability_distribution = theano.function([x, y, y_position], dist)
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

        self.get_batch_cost_unregularized = theano.function([docs, summaries], cost,
                                                allow_input_downcast=True)
        cost = cost + regularization_cost

        params = {p.name: p for p in self.params} 
        grads = T.grad(cost, self.params)

        # learning rate
        lr = T.scalar(name='lr')
        gradient_update, network_update = optimisers.sgd(lr, self.params,
                                                                grads, docs, summaries, cost)  
        return gradient_update, network_update

    def initialize(self, batch_sz):
        self.gradient_update, self.network_update = self.train_model_func(batch_sz)

    def train_one_batch(self, documents, summaries, learning_rate, verbose = False):
        cost = self.gradient_update(documents, summaries)
        self.network_update(learning_rate)

        if verbose:
            print("Cost after update: ", cost)

        return cost



