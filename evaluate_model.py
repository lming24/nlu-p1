import tensorflow as tf
import datasets
import preprocessing as prep
import constants as con
import datetime
import numpy as np

evaluation_model = "C"

meta_graph_path = './snapshots/Experiment_{}-240000.meta'.format(evaluation_model)
checkpoint_folder = './snapshots/Experiment_{}-240000'.format(evaluation_model)

output_file = open( "group10.perplexity{}".format(evaluation_model),'w')

def calculate_perplexities(losses, input_y, padding_id):

    num_sentences, len_sen = input_y.shape
    perplexities = []
    for i in range(num_sentences):
        s = (input_y[i] != padding_id)
        l_i = losses[i][s]
        perplexity = np.exp(np.mean(l_i))
        perplexities.append(perplexity)
    return perplexities


tf.reset_default_graph()

with tf.Session() as sess:

    saver = tf.train.import_meta_graph(meta_graph_path)
    saver.restore(sess, checkpoint_folder)


    graph = tf.get_default_graph()
    graph_input_y = graph.get_tensor_by_name("input_y:0")
    graph_input_x = graph.get_tensor_by_name("input_x:0")
    graph_batch_size = graph.get_tensor_by_name("batch_size:0")

    graph_predictions = graph.get_tensor_by_name("output_layer/predictions:0")
    graph_losses = graph.get_tensor_by_name("output_layer/losses/Reshape_2:0")

    nlpdata = datasets.NLUProject1Dataset()
    nlpdatatest = datasets.NLUProject1TestDataset()
    print(len(nlpdatatest.test_x))
    batches = prep.batch_iter(list(zip(nlpdatatest.test_x, nlpdatatest.test_y)), con.BATCH_SIZE, con.n_epochs)
    #print(len(batches))
    for batch in batches:
        x_batch, y_batch = zip(*batch)

        feed_dict = {
            graph_input_x: x_batch,
            graph_input_y: y_batch,
            graph_batch_size: len(x_batch)
        }

        losses, input_y = sess.run(
            [graph_losses, graph_input_y],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()

        perplexities = calculate_perplexities( losses, input_y, nlpdata.word_to_id[con.PAD])
        for perplexity in perplexities:
            output_file.write(str(perplexity) + '\n')

    output_file.close()