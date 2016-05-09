import network
import util
import numpy as np

#np.set_printoptions(threshold='nan')

def validate(net, input_batch, summary_batch):
    cost = net.get_batch_cost_unregularized(input_batch, summary_batch)
    print("Validation cost:", cost)
    return cost


def train_model(epochs, batch_size, save_params_every = 10, validate_every=10):
    print("Loading data...")
    dp = util.GWDataProcessor(load_from_file=True)
    dp.load_tensors()
    #dp.load_json_from_folder(data_folder)

    dp.calculate_network_parameters()

    num_batches = dp.get_num_batches(batch_size)

    print("Calculating network parameters...")


    embedding_size = 50

    emb_matrix = util.load_embeddings_from_glove(embedding_size, dp.vectorizer)

    input_sentence_length = dp.input_sentence_length
    context_length = dp.pad_length

    print("Building network...")

    s = network.SummarizationNetwork(vocab_size = dp.vectorizer.vocab_size() + 1,
            context_size = 3, input_sentence_length = input_sentence_length,
            embedding_size = embedding_size, embedding_matrix = emb_matrix)
    #grad_shared, update = s.initialize()

    params = s.params
    cpd = s.f_conditional_probability_distribution
    embedding_normalization_function = s.normalize_embeddings_func()


    ex_idx = 1 

    # validation inputs and summaries...
    v_summaries, v_inputs = dp.get_tensors_for_batch(num_batches-1, batch_size=batch_size)

    v_summaries, v_inputs = v_summaries.astype('int32'), v_inputs.astype('int32')

    for epoch_id in range(epochs):

        # for the moment... save last batch for validation
        for batch_id in range(num_batches-1):
            summaries, inputs = dp.get_tensors_for_batch(batch_id, batch_size=batch_size)

            s.train_one_batch(inputs, summaries, 0.1, True)

            inpt = inputs[ex_idx, :].astype('int32')
            summ = summaries[ex_idx, :].astype('int32') # was doing .A1 before...

            if batch_id % save_params_every == 0:
                s.save("yohgnet_{}_{}.network".format(epoch_id, batch_id))

            if batch_id % validate_every == 0:
                validate(s, v_inputs, v_summaries)

            if epoch_id < 1 or batch_id % validate_every != 0: continue

            rm = dp.vectorizer.generate_reverse_mapping()

            for idx in range(3,25):
                inpt_words = []

                summ_words = [rm[tk] for tk in summ]
                summ_so_far = summ_words[:idx]

                print(idx, summ_so_far)

                dist = cpd(inpt, summ, idx)

                indices = np.argsort(dist.T)[:10]
                print(indices)
                for index in indices[0][-10:]:
                    print(dist[index],rm[index])
                correct_word_token = summ_words[idx]
                correct_word_idx = dp.vectorizer.mapping[correct_word_token]
                print("correct word score:", dist[correct_word_idx], correct_word_token)

        #embedding_normalization_function()
 


if __name__ == "__main__":
    epochs = 15
    batch_size = 128
    train_model(data_folder, epochs, batch_size)
