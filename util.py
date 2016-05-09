import vectorizer
import os
import json
import numpy as np
import network
import gzip
import xml.etree.ElementTree as ET

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
    with open("glove.6B/glove.6B.{}d.txt".format(dimension), 'rb') as embfile:
        for i, line in enumerate(embfile, 1):
            try:
                line = line.decode('utf-8')
                line = line.split(" ")
                vec = list(map(float, line[1:]))
                glove_embeddings[line[0]] = vec
                #print(line[0], vec)
                assert(len(vec) == dimension)
            except UnicodeEncodeError as e:
                print("Decode error at {}".format(i))
        print("Total vectors loaded:{}".format(i))

    vocab_size = vectorizer.vocab_size()
    print("vocab size: {}".format(vocab_size))
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

class GWDataProcessor(object):
    def __init__(self, load_from_file = False):
        vparams = {"preserve_case": False}
        self.vectorizer = vectorizer.TextVectorizer(vparams) 
        self.pad_token = "PAD"
        self.end_token = "endtok"
        self.end_token_value = self.vectorizer.vectorize(self.end_token)[0]
        self.pad_length = 3
        self.summaries = []
        self.inputs = []
        self.num_files_loaded = 0

        self.last_file_idx = 0

        self.num_to_store_per_file = 100000

        self.gw_folder_path = "/afs/ir/data/linguistic-data/ldc/LDC2007T07_English-Gigaword-Third-Edition/data"
        self.valid_corpora = ["nyt_eng"]

        self.input_max_length = 0
        self.summary_max_length= 0

        self.num_pairs = 0

        self.data_loaded = False

        self.load_file_list()
        if load_from_file:
            self.load_metadata()

    def calculate_network_parameters(self):
        self.input_sentence_length = self.input_max_length 

    def load_file_list(self):
        self.data_files = []
        for corpora in self.valid_corpora:
            folder_path = os.path.join(self.gw_folder_path, corpora)
            files_in_folder = [os.path.join(self.gw_folder_path, corpora, f) for f in os.listdir(folder_path)]
            self.data_files.extend(files_in_folder)

    def save_metadata(self):
        metadata = {
                    'input_length': self.input_max_length,
                    'summary_length': self.summary_max_length,
                    'num_pairs': self.num_pairs
                   }
        with open('./data/metadata.json', 'w') as jf:
            json.dump(metadata, jf)
        self.vectorizer.save_mapping()

    def load_metadata(self):
        with open('./data/metadata.json', 'r') as jf:
            md = json.load(jf)
            self.input_max_length = md['input_length']
            self.summary_max_length = md['summary_length']
            self.num_pairs = md['num_pairs']
        self.vectorizer.load_mapping()

    def convert_to_tensors(self):
        pairs = []
        for filename in self.data_files:
            pairs.extend(self.convert_single_file_to_tensor(filename))
            if len(pairs) > self.num_to_store_per_file:
                to_store = pairs[:self.num_to_store_per_file]
                pairs = pairs[self.num_to_store_per_file:]
                self.dump_to_json(to_store)
        self.dump_to_json(pairs)

    def dump_to_json(self, pairs):
        with open('./data/{}.json'.format(self.last_file_idx), 'w') as jf:
            print("saving file {}".format(self.last_file_idx))
            json.dump(pairs, jf)
            self.last_file_idx += 1
        self.save_metadata()

    def convert_single_file_to_tensor(self, file_path):
        pairs = []
        with gzip.open(file_path, 'rb') as f:
            content = f.read()
            content = "<file>\n{}\n</file>".format(content)
            content = content.replace("&", "&amp;")
            root = ET.fromstring(content)
            for doc in root.findall("DOC"): 
                headline = doc.find("HEADLINE")
                if headline is None: continue
                headline = remove_escaped(headline.text)
                text = doc.find("TEXT")
                paragraphs = text.findall("P")
                first_three_paras = [p.text for p in paragraphs[:2]]
                first_three_paras = ' '.join(first_three_paras)
                first_three_paras = remove_escaped(first_three_paras)
                v_input = self.vectorizer.vectorize(first_three_paras)
                v_summary = self.vectorizer.vectorize(headline)
                self.input_max_length = max(self.input_max_length, len(v_input))
                self.summary_max_length = max(self.summary_max_length, len(v_summary))
                self.num_pairs += 1
                pair = DataPair(headline, first_three_paras, v_summary, v_input)
                pairs.append(pair)
        return [p.__dict__ for p in pairs]

    def get_batch_size(self, num_batches):
        return int( self.num_pairs / num_batches)

    def get_num_batches(self, batch_size):
        return int( self.num_pairs / batch_size)

    def load_tensors(self, text_cutoff_length=100):
        self.summaries = []
        self.indices = []
        file_idx = 0
        while True:
            try: 
                print("trying to load file... {}".format(file_idx))
                with open("data/{}.json".format(file_idx), 'r') as f:
                    data = json.load(f) 
                    print(len(data))
                    hlv = [d['headline_vec'] for d in data]
                    if text_cutoff_length is None:
                        tv = [d['text_vec'] for d in data]
                    else:
                        tv = [d['text_vec'][:text_cutoff_length] for d in data]
                    self.summaries.extend(hlv)
                    self.inputs.extend(tv)
                    print("current size: ", len(self.summaries))
            except FileNotFoundError as e:
                break
            file_idx += 1

        if text_cutoff_length is not None:
            self.input_max_length = text_cutoff_length

        print("Summary max len: {}".format(self.summary_max_length))
        print("Text max len: {}".format(self.input_max_length))

        #self.summaries = np.array(self.summaries)
        #self.inputs = np.array(self.inputs)

    def get_tensors_for_batch(self, batch_id, num_batches=None, batch_size=None, pad_length = None):
        num_examples = self.num_pairs 
        if ((batch_size is None and num_batches is None) or  
                (batch_size is not None and num_batches is not None)):
            raise Exception("Only one of batch_size or num_batches must be non-None")
        if num_batches is not None: batch_size = int(num_examples / num_batches)
        elif batch_size is not None: num_batches = int(num_examples / batch_size)

        start, end = batch_id * batch_size, (batch_id + 1) * batch_size
        summs = pad_list_of_indices(self.summaries[start:end], self.end_token_value, self.summary_max_length)
        texts = pad_list_of_indices(self.inputs[start:end], self.end_token_value, self.input_max_length)

        return np.array(summs), np.array(texts)
    
        

class DataPair(object):
    def __init__(self, headline, text, hl_vec, text_vec):
        self.headline = headline
        self.text = text
        self.headline_vec = hl_vec
        self.text_vec = text_vec

def remove_escaped(s):
    rep = {
        "\\n": " ",
        "\\'": "'",
        '\\"': '"'
        }
    for k, v in rep.items():
        s = s.replace(k, v)
    return s.strip()
                

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
    dp = GWDataProcessor()
    dp.convert_to_tensors()
    dp.save_metadata()
    exit()


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

