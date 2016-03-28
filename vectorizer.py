from nltk.tokenize import StanfordTokenizer, TweetTokenizer
import scipy.sparse as sp
import numpy as np

class TextVectorizer(object):
    def __init__(self, params, mapping = None):
        self.params = params
        if mapping is None:
            self.mapping = {}
        else:
            self.mapping = mapping
            # do some other stuff to restore mappings... TODO

        self.tokenizer = TweetTokenizer(preserve_case = self.params["preserve_case"]) 
        self.last_unassigned_index = 0

    def generate_reverse_mapping(self):
        rm = {}
        for token, val in self.mapping.items():
            rm[val] = token
        return rm 

    def vectorize(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        vectorized_text = []
        for token in tokenized_text:
            if token not in self.mapping.keys():
                self.mapping[token] = self.last_unassigned_index
                v = self.last_unassigned_index
                self.last_unassigned_index += 1
            else:
                v = self.mapping[token]
            vectorized_text.append(v)
        self.index_vector_to_sparse_matrix(vectorized_text) 
        return vectorized_text

    def vocab_size(self):
        return self.last_unassigned_index

    def index_vector_to_sparse_matrix(self, index_vector):
        data = np.asarray(len(index_vector) * [1])
        indices = np.asarray(index_vector)
        indptr = np.arange(0, len(index_vector) + 1)
        rows = self.vocab_size()
        cols = len(index_vector)
        m = sp.csc_matrix((data, indices, indptr), shape=(rows, cols))
        return m
if __name__ == "__main__":
    params = {"preserve_case": False}
    tv = TextVectorizer(params)

    input = "An atom is the smallest constituent unit of ordinary matter that has the properties \
            of a chemical element. Every solid, liquid, gas and plasma is composed of neutral or \
            ionized atoms"
    summary = "An atom is the smallest unit of matter"

    print(tv.vectorize(input))

    print(tv.vectorize(summary))

