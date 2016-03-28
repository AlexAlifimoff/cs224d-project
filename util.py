import vectorizer
import os
import json
import numpy as np
import network

def max_dimensions(matrices):
    dims = (0, 0)
    for m in matrices:
        dims = (max(dims[0], m.shape[0]), max(dims[1], m.shape[1]))
    return dims

def maxlen(lists):
    return len(max(lists, key = lambda x: len(x)))

def pad_list_of_matrices(matrices):
    padded = []
    max_dims = max_dimensions(matrices)
    return [np.pad(m, [(0, max_dims[0] - m.shape[0]), (0, max_dims[1] - m.shape[1])],
                    'constant', constant_values=(0,0)) for m in matrices]

def pad_list_of_indices(matrices, endtok):
    padded = []
    max_dims = maxlen(matrices)
    print("max len:", max_dims)
    print(matrices[0])
    return [m + (max_dims - len(m)) * [endtok] for m in matrices ]

class DataProcessor(object):
    def __init__(self):
        vparams = {"preserve_case": False}
        self.vectorizer = vectorizer.TextVectorizer(vparams) 
        self.pad_token = "PAD"
        self.pad_length = 3
        self.summaries = []
        self.inputs = []

    def load_json_from_folder(self, folder_path):
        files_in_folder = [f for f in os.listdir(folder_path)
                            if os.path.isfile(os.path.join(folder_path, f))]
        print("Loading files...")
        print(files_in_folder)

        for f in files_in_folder:
            self.load_single_json(os.path.join(folder_path, f))

    def load_single_json(self, file_path):
        with open(file_path) as json_file:
            print(file_path)
            try:
                data = json.load(json_file) 
                print(data)
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

    def get_tensors_for_batch(self, batch_id, num_batches):
        num_examples = len(self.summaries)
        batch_size = int(num_examples / num_batches)
        summ_matrices = []
        text_matrices = []
        vec_to_sm = self.vectorizer.index_vector_to_sparse_matrix
        endtok = self.vectorizer.vocab_size()
        self.vectorizer.vectorize("endtok")
        start, end = batch_id * batch_size, (batch_id + 1) * batch_size
        summs = pad_list_of_indices(self.summaries[start:end], endtok)
        texts = pad_list_of_indices(self.inputs[start:end], endtok)

        self.input_sentence_length = len(texts[0])

        for summ, text in zip(summs, texts):
            summ_matrices.append(vec_to_sm(summ).toarray())
            text_matrices.append(vec_to_sm(text).toarray())

        return np.stack(summ_matrices, axis = 0), np.stack(text_matrices, axis = 0)

if __name__ == "__main__":
    dp = DataProcessor()
    dp.load_json_from_folder("../cs224n-project/data/wikipedia")

    print( len(dp.summaries), len(dp.inputs) )
    print(dp.summaries[-1])
    print(dp.inputs[-1])

    print( dp.vectorizer.vocab_size() )
    summary_tensor, input_tensor = dp.get_tensors_for_batch(0, 1000) 

    context_length = dp.pad_length
    input_sentence_length = dp.input_sentence_length

    s = network.SummarizationNetwork(vocab_size = dp.vectorizer.vocab_size(), context_size = context_length,
           input_sentence_length = input_sentence_length)
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

    dist = cpd(input_tensor[ex_idx, : , :], summary_tensor[ex_idx, :, :], idx)

    rm = dp.vectorizer.generate_reverse_mapping()
    indices = np.argsort(dist.T)[:25]
    print(indices)
    for index in indices[0][-50:]:
        print(index)
        print(dist[index], rm[index])
    #for i in range(dist.shape[0]):
    #    print(dist[i, 0], rm[i])

