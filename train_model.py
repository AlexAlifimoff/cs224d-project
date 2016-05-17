import network
import util
import numpy as np
import baseline_logreg
import k_baseline_logreg

#np.set_printoptions(threshold='nan')

def validate(net, input_batch, summary_batch, imasks, smasks):
    cost = net.get_batch_cost_unregularized(input_batch, summary_batch, imasks, smasks)
    print("Validation cost:", cost)
    return cost


def train_model(epochs, batch_size, save_params_every = 10, validate_every=10, batches_per_update=1, model = "BaselineLR"):
    print("Loading data...")
    dp = util.GWDataProcessor(load_from_file=True)
    dp.load_tensors()
    #dp.load_json_from_folder(data_folder)

    dp.calculate_network_parameters()

    num_batches = dp.get_num_batches(batch_size*batches_per_update)

    print("Calculating network parameters...")
    embedding_size = 50

    emb_matrix = util.load_embeddings_from_glove(embedding_size, dp.vectorizer) #np.random.rand( embedding_size, dp.vectorizer.vocab_size() )#

    input_sentence_length = dp.input_sentence_length
    context_length = dp.pad_length

    print("Building network...")
    if model == "Full":
        s = network.SummarizationNetwork(vocab_size = dp.vectorizer.vocab_size() + 1,
            context_size = context_length, input_sentence_length = input_sentence_length,
            embedding_size = embedding_size, embedding_matrix = emb_matrix, batch_size = batch_size,
            summary_length = dp.summary_max_length, num_batches = batches_per_update)

    elif model == "BaselineLR":
        s = k_baseline_logreg.LogRegBaseline(context_length, embedding_size, emb_matrix,
            input_sentence_length, dp.vectorizer.vocab_size(), dp.summary_max_length, batch_size,
            num_batches = batches_per_update, eos_token = dp.vectorizer.EOS_value)
    print("Done building network")

    params = s.params
    for p in params: print(p.get_value())
    cpd = s.f_conditional_probability_distribution
    if model == "Full":
        embedding_normalization_function = s.normalize_embeddings_func()


    ex_idx = 1 

    # validation inputs and summaries...
    v_summaries, v_inputs, vs_masks, vi_masks = dp.get_tensors_for_batch(num_batches-1, batch_size=batch_size)

    v_summaries, v_inputs = v_summaries.astype('int32'), v_inputs.astype('int32')

    best_val_cost = np.inf
    lr = 0.0001

    for epoch_id in range(epochs):
        # for the moment... save last batch for validation
        print("EPOCH ID", epoch_id)
        for batch_id in range(num_batches-1):
            summaries, inputs, smasks, imasks = dp.get_tensors_for_batch(batch_id, batch_size=batch_size*batches_per_update)

            print(summaries.shape, inputs.shape)

            s.train_one_superbatch(inputs, summaries, imasks, smasks, lr, batches_per_update)

            inpt = inputs[ex_idx, :].astype('int32')
            summ = summaries[ex_idx, :].astype('int32') # was doing .A1 before...
            inpt_ = np.expand_dims(inpt, axis=0)
            summ_ = np.expand_dims(summ, axis=0) 

            if batch_id % save_params_every == 0:
                #s.save("{}_{}_{}.network".format(s.__class__.__name__, epoch_id, batch_id))
                pass

            if batch_id % validate_every == 0:
                if model == 'BaselineLR':
                    s.validate(v_inputs, v_summaries)
                else:   
                    vc = validate(s, v_inputs, v_summaries, vi_masks, vs_masks)
                    if batch_id == 0:
                        if vc > best_val_cost:
                           print("halving learning rate")
                           lr = lr / 2.0
                        else:
                            print("updating best val cost")
                            best_val_cost = vc

            if epoch_id < 0 or batch_id % validate_every != 0: continue

            rm = dp.vectorizer.generate_reverse_mapping()

            for idx in range(3,25):
                inpt_words = []

                summ_words = [rm[tk] for tk in summ]
                summ_so_far = summ_words[:idx]

                if summ_so_far[-1] == '<eos>': break
                print(idx, summ_so_far)


                dist = cpd(inpt_, summ_, idx).T
                print("dist\n", dist)
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
    train_model(epochs, batch_size)
