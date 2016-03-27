from network import *
from vectorizer import *
import theano.tensor as T
import theano
import optimisers

np.set_printoptions(threshold=np.nan)

def max_dimensions(matrices):
    dims = (0, 0)
    for m in matrices:
        dims = (max(dims[0], m.shape[0]), max(dims[1], m.shape[1]))
    return dims

def pad_list_of_matrices(matrices):
    padded = []
    max_dims = max_dimensions(matrices)
    return [np.pad(m, [(0, max_dims[0] - m.shape[0]), (0, max_dims[1] - m.shape[1])],
                    'constant', constant_values=(0,0)) for m in matrices]

params = {"preserve_case": False}
tv = TextVectorizer(params)

context_length = 3

input = "An atom is the smallest constituent unit of ordinary matter that has the properties \
         of a chemical element. Every solid, liquid, gas and plasma is composed of neutral or \
         ionized atoms"
summary = "An atom is the smallest unit of matter"

input2 = "Connor Van Gessel is my roomate. He was born in Marin County and sometimes likes to \
          clip the nails of our cats. He is a good person."
summary2 = "Connor Van Gessel is my roomate, likes cats, and is a good person."

input3 = "The Kansas Jayhawks are a basketball team representing the University of Kansas. \
          They are located in Lawrence, Kansas and their stadium is Allen Fieldhouse."
summary3 = "The Kansas Jayhawks are a basketball team."

summary = ''.join(["PAD "] * context_length) + summary
summary2 = ''.join(["PAD "] * context_length) + summary2
summary3 = ''.join(["PAD "] * context_length) + summary3

summaries = [summary, summary2, summary3]
inputs = [input, input2, input3]

v_summaries = [tv.vectorize(i) for i in summaries]
v_inputs = [tv.vectorize(i) for i in inputs]

sm_inputs = [tv.index_vector_to_sparse_matrix(i).toarray() for i in v_inputs]
sm_summs = [tv.index_vector_to_sparse_matrix(i).toarray() for i in v_summaries]

padded_inputs = pad_list_of_matrices(sm_inputs)
padded_summaries = pad_list_of_matrices(sm_summs)

input_tensor = np.stack(padded_inputs, axis = 0)
summary_tensor = np.stack(padded_summaries, axis = 0)
input_sentence_length = input_tensor.shape[2] 

print(input_sentence_length)
s = SummarizationNetwork(vocab_size = tv.vocab_size(), context_size = context_length,
           input_sentence_length = input_sentence_length)
#s.load('yohgnet.network')
print("Network built.")

print("building func...")
grad_shared, update = s.initialize()
cpd = s.f_conditional_probability_distribution
idx = 1 
idx += context_length
ex_idx = 0
print(cpd(padded_inputs[ex_idx], padded_summaries[ex_idx], idx))

for i in range(200):
    cost = grad_shared(input_tensor, summary_tensor)
    update(0.05)
    #print(cost)
print("update done")

print("cpd...")
dist = cpd(padded_inputs[ex_idx], padded_summaries[ex_idx], idx)

rm = tv.generate_reverse_mapping()
for i in range(dist.shape[0]):
    print(dist[i, 0], rm[i])
print(inputs[0])
print(summaries[0])

theano.printing.pydotprint(cpd, outfile='network_graph.png', var_with_name_simple=True)
#s.save('yohgnet.network')
