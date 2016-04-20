import network
import util
import numpy as np

#np.set_printoptions(threshold='nan')

def train_model(data_folder, epochs, batch_size):
    print("Loading data...")
    dp = util.DataProcessor()
    dp.load_json_from_folder(data_folder)

    num_batches = dp.get_num_batches(batch_size)

    print("Calculating network parameters...")

    dp.calculate_network_parameters()

    embedding_size = 50

    emb_matrix = util.load_embeddings_from_glove(embedding_size, dp.vectorizer)

    input_sentence_length = dp.input_sentence_length
    context_length = dp.pad_length

    print("Building network...")

    s = network.SummarizationNetwork(vocab_size = dp.vectorizer.vocab_size() + 1,
            context_size = 3, input_sentence_length = input_sentence_length,
            embedding_size = embedding_size, embedding_matrix = emb_matrix)
    grad_shared, update = s.initialize()

    params = s.params
    cpd = s.f_conditional_probability_distribution
    embedding_normalization_function = s.normalize_embeddings_func()


    ex_idx = 1 


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
            update(0.1)

            print(epoch_id, batch_id, cost)

            inpt = inputs[ex_idx, :].A1.astype('int32')
            summ = summaries[ex_idx, :].A1.astype('int32')
            print(inpt)

            if epoch_id < 0: continue

            E = [e for e in params if e.name == 'V'][0]
            print("Embedding matrix...")
            print(E.get_value())
            #emb_matrix = E.get_value()


            rm = dp.vectorizer.generate_reverse_mapping()
            #print("endtok vector")
            #print(emb_matrix[:, dp.vectorizer.mapping["endtok"]])

            



            for idx in range(3, 10):

                inpt_words = []
                for i, tk in enumerate(inpt):
                    inpt_words.append( rm[tk] )
                    if i >= idx: break

                print(idx, inpt_words)

                dist = cpd(inpt, summ, idx)


                indices = np.argsort(dist.T)[:10]
                print(indices)
                for index in indices[0][-10:]:
                    print(dist[index],rm[index])

        #embedding_normalization_function()
 


if __name__ == "__main__":
    data_folder = "./data/nyt/data/nyt_eng/" #"../cs224n-project/data/wikipedia_small"
    epochs = 15
    batch_size = 128
    train_model(data_folder, epochs, batch_size)
