import optimisers
import theano.tensor as T
import theano
import numpy as np
import network

from six.moves import cPickle as pickle
import os

class LogRegBaseline(object):
    def __init__(self, context_size, embedding_size, embedding_matrix, input_length,
                    vocab_sz, summary_length, batch_sz, num_batches = 10):
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.input_length = input_length
        self.vocab_sz = vocab_sz
        self.summary_length = summary_length
        self.num_batches = num_batches

        self.initialize(batch_sz)

    def conditional_probability_distribution(self, x, y):
        self.context_embedding = network.EmbeddingsLayer(y, self.context_size, self.embedding_size,
                                    self.vocab_sz, True, 'E', self.embedding_matrix)
        self.input_embedding = network.EmbeddingsLayer(x, self.input_length, self.embedding_size,
                                    self.vocab_sz, True, 'F', self.embedding_matrix)

        self.word_vecs = T.concatenate([self.context_embedding.output, self.input_embedding.output])
        # size should be context_size * embedding_size + input_length * embedding_size

        input_sz = (self.context_size + self.input_length) * self.embedding_size

        V_value = np.random.uniform(low=-0.02, high=0.02, size=(self.vocab_sz, input_sz))
        self.V = theano.shared(
                    value = V_value.astype(theano.config.floatX),
                    name = 'V', 
                    borrow = True)

        b_value = np.random.uniform(low=-0.02, high=0.02, size=(self.vocab_sz, 1))

        self.b = theano.shared( value = b_value.astype(theano.config.floatX), name = 'b',
                        borrow = True)

        self.params = [self.V, self.b] + self.context_embedding.params + self.input_embedding.params

        return T.nnet.softmax((T.dot(self.V, self.word_vecs) + self.b).T).T


    def conditional_probability(self, x, y, y_position, xmasks, ymasks):
                
        dist = self.conditional_probability_distribution(x,
                y[y_position-self.context_size:y_position])

        self.f_conditional_probability_distribution = theano.function([x, y, y_position], dist)
        token = y[y_position]
        return -T.log(dist[token]) * ymasks[y_position]

    def negative_log_likelihood(self, x, y, xmasks, ymasks):
        # Here, y is an entire summary and x is an entire text.
        # We start at the first y, take its context and it, and then
        # calculate the probability of our current word appearing
        
        # Since we pad the input vectors, we start at C and go to len(y_original) + C
        y_position_range = T.arange(self.context_size, self.summary_length)

        # We start by indexing into y
        # current_word = y[:, i]
        # context = y[:, i-1-C:i-1]

        func = lambda y_pos, x, y, x_masks, y_masks: self.conditional_probability(x, y, y_pos, x_masks, y_masks)

        probabilities, _ = theano.scan(func, sequences=y_position_range,
                                    non_sequences = [x, y, xmasks, ymasks], n_steps=self.summary_length-self.context_size)

        return probabilities.T

    def negative_log_likelihood_batch(self, documents, summaries, docmasks, summasks, batch_size):
        document_range = T.arange(0, batch_size)
        func = lambda i, documents, summaries: T.sum(self.negative_log_likelihood(documents[i, :],
            summaries[i, :], docmasks[i, :], summasks[i, :]), axis=1)

        probs, _ = theano.scan(func, sequences=[document_range], non_sequences=[documents,summaries])
        return probs.sum() 

    def train_model_func(self, batch_size, num_batches, summary_sz, input_sz):
        summaries = T.imatrix('summaries')
        docs = T.imatrix('docs')
        docmasks = T.imatrix('docmasks')
        summasks = T.imatrix('summasks')

        s = np.zeros((batch_size * num_batches, summary_sz))
        d = np.zeros((batch_size * num_batches, input_sz))

        sm = np.zeros((batch_size * num_batches, summary_sz))
        dm = np.zeros((batch_size * num_batches, input_sz))



        summary_superbatch = theano.shared( s.astype(theano.config.floatX),
                                           name = 's_summs', borrow = True )
        doc_superbatch = theano.shared( d.astype(theano.config.floatX),
                                        name = 's_docs', borrow = True )
        docm_superbatch = theano.shared( dm.astype(theano.config.floatX),
                                        name = 's_docmasks', borrow = True )
        summarym_superbatch = theano.shared( sm.astype(theano.config.floatX),
                                        name = 's_summasks', borrow = True )



        self.ssb = summary_superbatch
        self.dsb = doc_superbatch
        self.dmsb = docm_superbatch
        self.smsb = summarym_superbatch

        cost = self.negative_log_likelihood_batch(docs, summaries, docmasks, summasks, batch_size)
        #regularization_cost = self.l2_coefficient * sum([(p ** 2).sum() for p in self.params])

        self.get_batch_cost_unregularized = theano.function([docs, summaries, docmasks, summasks], cost,
                                                 allow_input_downcast=True)
        #theano.printing.debugprint(cost)
        #cost = cost + regularization_cost

        params = {p.name: p for p in self.params} 
        grads = T.grad(cost, self.params)
        #grads = theano.printing.Print("grads")(grads)

        # learning rate
        lr = T.scalar(name='lr')
        gradient_update = optimisers.sgd__(lr, self.params, grads, docs, summaries, docmasks, summasks,
                                        cost, self.dsb, self.ssb, self.dmsb, self.smsb, batch_size)
        return gradient_update

    def train_one_superbatch(self, dsb, ssb, dmsb, smsb, learning_rate, num_batches):
        #shared_docs, shared_summs = shared_dataset(documents, summaries)
        self.dsb.set_value(dsb)
        self.ssb.set_value(ssb)
        #print(dmsb)
        #dmsb = dmsb.astype(int)
        #smsb = smsb.astype(int)
        self.dmsb.set_value(dmsb)
        self.smsb.set_value(smsb)
        print("setting shared variables...")
        cost = 0
        for i in range(num_batches):
            c = self.gradient_update(learning_rate, i)
            print(c)
            cost += c
            #self.network_update(learning_rate, i)

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

    def initialize(self, batch_sz):
        self.gradient_update = self.train_model_func(batch_sz, self.num_batches, self.summary_length,
                                    self.input_length)

    def train_one_batch(self, documents, summaries, learning_rate, verbose = False):
        cost = self.gradient_update(documents, summaries)
        self.network_update(learning_rate)

        if verbose:
            print("Cost after update: ", cost)

        return cost



