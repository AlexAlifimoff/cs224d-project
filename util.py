import vectorizer
import os
import json
import numpy as np
import network

def max_dimensions(matrices):
    if isinstance(matrices[0], list):
        return max([len(m) for m in matrices])
    dims = (0, 0)
    for m in matrices:
        dims = (max(dims[0], m.shape[0]), max(dims[1], m.shape[1]))
    return dims

def maxlen(lists):
    return len(max(lists, key = lambda x: len(x)))

def pad_list_of_matrices(matrices, max_dims = None):
    padded = []
    if max_dims is None: max_dims = max_dimensions(matrices)
    return [np.pad(m, [(0, max_dims[0] - m.shape[0]), (0, max_dims[1] - m.shape[1])],
                    'constant', constant_values=(0,0)) for m in matrices]

def pad_list_of_indices(matrices, endtok, pad_length = None):
    padded = []
    if pad_length is None: max_dims = maxlen(matrices)
    else: max_dims = pad_length
    #print("max len:", max_dims)
    #print(matrices[0])
    return [m + (max_dims - len(m)) * [endtok] for m in matrices ]

def load_embeddings_from_glove(dimension, vectorizer):
    valid_dimensions = [50, 100, 200, 300]
    assert(dimension in valid_dimensions)

    glove_embeddings = {}
    with open("glove.6B/glove.6B.{}d.txt".format(dimension), 'r') as embfile:
        for line in embfile:
            line = line.split(" ")
            vec = list(map(float, line[1:]))
            glove_embeddings[line[0]] = vec
            assert(len(vec) == dimension)


    vocab_size = vectorizer.vocab_size()
    mapping = vectorizer.mapping

    # embedding size by vocab size
    emb_matrix = np.zeros( (dimension, vocab_size) )

    indices_to_replace = []
    for word, index in mapping.items():
        try:
            vec = glove_embeddings[word]
        except KeyError as e:
            indices_to_replace.append(index)

        emb_matrix[:, index] = np.array(vec)

    avg_vector = np.mean(emb_matrix, axis = 1)

    for index in indices_to_replace:
        emb_matrix[:, index] = avg_vector

    print("Number of indices replaced with average:", len(indices_to_replace))

    return emb_matrix

class DataProcessor(object):
    def __init__(self):
        vparams = {"preserve_case": False}
        self.vectorizer = vectorizer.TextVectorizer(vparams) 
        self.pad_token = "PAD"
        self.end_token = "endtok"
        self.end_token_value = self.vectorizer.vectorize(self.end_token)[0]
        self.pad_length = 3
        self.summaries = []
        self.inputs = []
        self.num_files_loaded = 0

    def load_json_from_folder(self, folder_path):
        files_in_folder = [f for f in os.listdir(folder_path)
                            if os.path.isfile(os.path.join(folder_path, f))]
        print("Loading files...")
        print("Num loaded:", len(files_in_folder))

        for f in files_in_folder:
            try:
                self.load_single_json(os.path.join(folder_path, f))
                self.num_files_loaded += 1
            except json.decoder.JSONDecodeError as err:
                pass

    def load_single_json(self, file_path):
        with open(file_path) as json_file:
            try:
                data = json.load(json_file) 
                #print(data)
                summary = data["summary"]
                summary = ''.join([self.pad_token + " "] * self.pad_length) + summary
                full_text = data["full_text"]
                full_text = ''.join([self.pad_token + " "] * self.pad_length) + full_text
                vectorized_summary = self.vectorizer.vectorize(summary)
                vectorized_ft = self.vectorizer.vectorize(full_text)
                self.summaries.append(vectorized_summary)
                self.inputs.append(vectorized_ft)
            except json.decoder.JSONDecodeError as err:
                print("decode error.")
                raise err

    def get_batch_size(self, num_batches):
        return int( len(self.summaries) / num_batches)

    def get_num_batches(self, batch_size):
        return int( len(self.summaries) / batch_size)

    def calculate_network_parameters(self):
        summary_dims = max_dimensions(self.summaries)
        input_dims = max_dimensions(self.inputs)

        self.summary_max_length = summary_dims
        self.input_max_length = input_dims

        self.input_sentence_length = input_dims

    def get_tensors_for_batch(self, batch_id, num_batches=None, batch_size=None, pad_length = None):
        #if self.input_sentence_length is not None and pad_length is None:
        #    pad_length = self.input_sentence_length
        #    print("setting pad length to ", pad_length)
        num_examples = len(self.summaries)
        if ((batch_size is None and num_batches is None) or  
                (batch_size is not None and num_batches is not None)):
            raise Exception("Only one of batch_size or num_batches must be non-None")
        if num_batches is not None: batch_size = int(num_examples / num_batches)
        elif batch_size is not None: num_batches = int(num_examples / batch_size)

        vec_to_sm = self.vectorizer.index_vector_to_sparse_matrix
        #endtok = self.vectorizer.vocab_size()
        #self.vectorizer.vectorize("endtok")
        start, end = batch_id * batch_size, (batch_id + 1) * batch_size
        print(start, end)
        summs = pad_list_of_indices(self.summaries[start:end], self.end_token_value, self.summary_max_length)
        texts = pad_list_of_indices(self.inputs[start:end], self.end_token_value, self.input_max_length)

        #self.input_sentence_length = len(texts[0])
        #print(type(texts))
        #print(texts)
        #print(type(texts[0]))
        #for t in texts:
        #    print(len(t))
        return np.matrix(summs), np.matrix(texts)

if __name__ == "__main__":


    dp = DataProcessor()
    dp.load_json_from_folder("../cs224n-project/data/wikipedia")

    print( len(dp.summaries), len(dp.inputs) )
    print(dp.summaries[-1])
    print(dp.inputs[-1])

    print( dp.vectorizer.vocab_size() )


    emb_matrix = load_embeddings_from_glove(50, dp.vectorizer)
    print(emb_matrix)

    summary_tensor, input_tensor = dp.get_tensors_for_batch(0, 1000) 

    context_length = dp.pad_length
    input_sentence_length = dp.input_sentence_length

    s = network.SummarizationNetwork(vocab_size = dp.vectorizer.vocab_size(), context_size = context_length,
           input_sentence_length = input_sentence_length, embedding_matrix = embedding_matrix)
    grad_shared, update = s.initialize()
    cpd = s.f_conditional_probability_distribution
    idx = 1 
    idx += context_length
    ex_idx = 0

    for i in range(20):
        cost = grad_shared(input_tensor, summary_tensor)
        update(0.05)
        print(cost)
    print("update done")

    inpt = input_tensor[ex_idx, :].A1.astype('int32')
    summ = summary_tensor[ex_idx, :].A1.astype('int32')

    dist = cpd(inpt, summ, idx)

    rm = dp.vectorizer.generate_reverse_mapping()
    indices = np.argsort(dist.T)[:25]
    print(indices)
    for index in indices[0][-50:]:
        print(index)
        print(dist[index], rm[index])
    #for i in range(dist.shape[0]):
    #    print(dist[i, 0], rm[i])

