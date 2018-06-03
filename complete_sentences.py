import tensorflow as tf
import datasets
import preprocessing as prep
import constants as con
import datetime
import numpy as np
from copy import deepcopy


nlpdata = datasets.NLUProject1Dataset()

def show_sentence(arr):
    sentence = []
    fin = next((index for index, value in enumerate(arr) if value == nlpdata.word_to_id[con.EOS] or value ==  nlpdata.word_to_id[con.PAD]), None)
    if arr[fin] == nlpdata.word_to_id[con.PAD]: fin -=1
    for i in arr:
        kk = nlpdata.id_to_word[i]
        sentence.append(kk)
    return ' '.join(sentence[1:fin+1])

meta_graph_path = './snapshots/Experiment_C-240000.meta'
checkpoint_folder = './snapshots/Experiment_C-240000'

tf.reset_default_graph()


with tf.Session() as sess:

    saver = tf.train.import_meta_graph(meta_graph_path)
    saver.restore(sess, checkpoint_folder)

    graph = tf.get_default_graph()
    graph_input_y = graph.get_tensor_by_name("input_y:0")
    graph_input_x = graph.get_tensor_by_name("input_x:0")
    graph_batch_size = graph.get_tensor_by_name("batch_size:0")

    graph_predictions = graph.get_tensor_by_name("output_layer/predictions:0")

    nlpdatasc = datasets.NLUProject1SentenceContinuationDataset()
    file = open("group10.continuation" , "w")


    for k, batch in enumerate(zip(nlpdatasc.sc_x, nlpdatasc.sc_x)):

        x_batch, y_batch = batch
        feed = deepcopy(x_batch)

        fin = next((index for index, value in enumerate(x_batch) if value == nlpdata.word_to_id[con.EOS]), None)
        feed[fin] = 0

        while feed[fin] != nlpdata.word_to_id[con.EOS] and fin < 21:

            feed_dict = {
                graph_input_x: np.array([feed]),
                graph_batch_size: 1
            }

            pred, input_x = sess.run(
                [graph_predictions, graph_input_x],
                feed_dict)

            feed[fin] = pred[0][fin-1]

            fin += 1

        time_str = datetime.datetime.now().isoformat()
        print("{}: predictions for sentence {}:\n {}\n {} ".format(time_str,k,show_sentence(feed),show_sentence(x_batch) ))
        file.write(show_sentence(feed) +"\n")

