from nltk.tokenize import TweetTokenizer
import scipy.sparse as sp
import numpy as np
import json

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class BetterVectorizer(object):
    def __init__(self):
        self.UNK = "<unk>"
        self.PAD = "<pad>"
        self.EOS = "<eos>"
        self.DIGIT = "<digit>"

        self.special_tokens = [self.UNK, self.PAD, self.EOS, self.DIGIT]

        self.mapping = {}
        self.UNK_value = self.mapping[self.UNK] = 0
        self.PAD_value = self.mapping[self.PAD] = 1
        self.EOS_value = self.mapping[self.EOS] = 2
        self.DIGIT_value = self.mapping[self.DIGIT] = 3

        self.last_unassigned_idx = 4 
        self.counts = {} 

        self.tokenizer = TweetTokenizer(preserve_case = False) 

        self.frequency_threshold = 5
        self.allow_digits = False

    def vocab_size(self):
        return self.last_unassigned_idx

    def count(self, text):
        tokenized = self.tokenizer.tokenize(text)
        for t in tokenized:
            if t in self.counts.keys():
                self.counts[t] += 1
            else:
                self.counts[t] = 1

    def generate_reverse_mapping(self):
        rm = {}
        for token, val in self.mapping.items():
            rm[val] = token
        return rm 


    def vectorize(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        vectorized_text = []
        for token in tokenized_text:
            if is_number(token): token = self.DIGIT
            else:
                count = self.counts[token]
                if count < self.frequency_threshold:
                    token = self.UNK
            if token not in self.mapping.keys():
                self.mapping[token] = self.last_unassigned_idx
                v = self.last_unassigned_idx
                self.last_unassigned_idx += 1
            else:
                v = self.mapping[token]
            vectorized_text.append(v)
        return vectorized_text

    def save_mapping(self):
        with open("mapping.json", 'w') as mf:
            json.dump(self.mapping, mf)
        with open("counts.json", 'w') as cf:
            json.dump(self.counts, cf)
        with open("vec_metadata.json", 'w') as mf:
            vmd = {
                'last_unassigned_index': self.last_unassigned_idx,
                #'params': self.params
                }
            json.dump(vmd, mf)

    def load_mapping(self):
        with open("mapping.json", 'r') as mf:
            self.mapping = json.load(mf)
        with open("counts.json", 'r') as cf:
            self.counts = json.load(cf)
        with open('vec_metadata.json', 'r') as mf:
            vmd = json.load(mf)
            self.last_unassigned_idx = vmd['last_unassigned_index']
            #self.params = vmd['params']



class TextVectorizer(object):
    def __init__(self, params, mapping = None):
        self.UNK = "UNK"
        self.params = params
        if mapping is None:
            self.mapping = {}
            self.counts = {}
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

    def vectorize(self, text, count_cutoff = None):
        tokenized_text = self.tokenizer.tokenize(text)
        vectorized_text = []
        for token in tokenized_text:
            if count_cutoff is None:
                if token not in self.mapping.keys():
                    self.mapping[token] = self.last_unassigned_index
                    v = self.last_unassigned_index
                    self.last_unassigned_index += 1
                    self.counts[token] = 1
                else:
                    v = self.mapping[token]
                    self.counts[token] += 1
            vectorized_text.append(v)
        #self.index_vector_to_sparse_matrix(vectorized_text) 
        return vectorized_text

    def vocab_size(self, count_cutoff = None):
        if count_cutoff is None: return self.last_unassigned_index
        else: 
            c = 0
            for count in self.counts.values():
                if count > count_cutoff:
                    c += 1
            return c

    def index_vector_to_sparse_matrix(self, index_vector):
        data = np.asarray(len(index_vector) * [1])
        indices = np.asarray(index_vector)
        indptr = np.arange(0, len(index_vector) + 1)
        rows = self.vocab_size()
        cols = len(index_vector)
        m = sp.csc_matrix((data, indices, indptr), shape=(rows, cols))
        return m

    def save_mapping(self):
        with open("mapping.json", 'w') as mf:
            json.dump(self.mapping, mf)
        with open("counts.json", 'w') as cf:
            json.dump(self.counts, cf)
        with open("vec_metadata.json", 'w') as mf:
            vmd = {
                'last_unassigned_index': self.last_unassigned_index,
                #'params': self.params
                }
            json.dump(vmd, mf)

    def load_mapping(self):
        with open("mapping.json", 'r') as mf:
            self.mapping = json.load(mf)
        with open("counts.json", 'r') as cf:
            self.counts = json.load(cf)
        with open('vec_metadata.json', 'r') as mf:
            vmd = json.load(mf)
            self.last_unassigned_index = vmd['last_unassigned_index']
            #self.params = vmd['params']

if __name__ == "__main__":
    params = {"preserve_case": False}
    tv = TextVectorizer(params)

    input = "An atom is the smallest constituent unit of ordinary matter that has the properties \
            of a chemical element. Every solid, liquid, gas and plasma is composed of neutral or \
            ionized atoms"
    summary = "An atom is the smallest unit of matter"

    print(tv.vectorize(input))

    print(tv.vectorize(summary))

