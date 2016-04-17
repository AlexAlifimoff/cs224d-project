import network
import util
import numpy as np

def train_model(data_folder, epochs, batch_size):
    dp = util.DataProcessor()
    dp.load_json_from_folder(data_folder)

    num_batches = dp.get_num_batches(batch_size)

    dp.calculate_network_parameters()

    input_sentence_length = dp.input_sentence_length
    context_length = dp.pad_length

    s = network.SummarizationNetwork(vocab_size = dp.vectorizer.vocab_size() + 1,
            context_size = context_length, input_sentence_length = input_sentence_length)
    grad_shared, update = s.initialize()
    cpd = s.f_conditional_probability_distribution

    idx = 1 
    idx += context_length
    ex_idx = 0


    for epoch_id in range(epochs):
        for batch_id in range(num_batches):
            summaries, inputs = dp.get_tensors_for_batch(batch_id, batch_size=batch_size)
            #print("inputs")
            #print(inputs)
            #print(type(inputs))
            #print(inputs.shape)
            #print("summaries")
            #print(summaries)
            #print(type(summaries))
            #print(summaries.shape)

            cost = grad_shared(inputs, summaries)
            update(0.001)

            print(epoch_id, batch_id, cost)

            inpt = inputs[ex_idx, :].A1.astype('int32')
            summ = summaries[ex_idx, :].A1.astype('int32')
            print(inpt)

            dist = cpd(inpt, summ, idx)

            rm = dp.vectorizer.generate_reverse_mapping()
            indices = np.argsort(dist.T)[:25]
            #print(indices)
            for index in indices[0][-50:]:
                #print(index)
                print(dist[index], rm[index])
 


if __name__ == "__main__":
    data_folder = "../cs224n-project/data/wikipedia"
    epochs = 10
    batch_size = 1000
    train_model(data_folder, epochs, batch_size)
