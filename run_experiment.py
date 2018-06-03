import tensorflow as tf
import constants as con
import LSTM_A
import LSTM_B
import LSTM_C
import numpy as np
import datetime
import datasets
import preprocessing as prep


def calculate_perplexities(losses, input_y, padding_id):

    num_sentences, len_sen = input_y.shape
    perplexities = []
    for i in range(num_sentences):
        s = (input_y[i] != padding_id)
        l_i = losses[i][s]
        perplexity = np.exp(np.mean(l_i))
        perplexities.append(perplexity)
    return perplexities


def train_step(lstm, x_batch, y_batch):

    feed_dict = {
        lstm.input_x: x_batch,
        lstm.input_y: y_batch,
        lstm.batch_size: len(x_batch)
    }
    _, step, loss, accuracy, losses, input_y = sess.run(
        [train_op, global_step, lstm.loss, lstm.accuracy, lstm.losses, lstm.input_y],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print(_)
    perplexity = calculate_perplexities(losses, input_y, nlpdata.word_to_id[con.PAD])
    avg_perplexity = sum(perplexity) / len(perplexity)

    #print("{}: step {}, loss {:g}, acc {:g}, perplexity {}".format(time_str, step, loss, accuracy, avg_perplexity))

    return loss, accuracy, perplexity


def dev_step(lstm, x_batch, y_batch):

    feed_dict = {
        lstm.input_x: x_batch,
        lstm.input_y: y_batch,
        lstm.batch_size: len(x_batch)
    }

    loss, accuracy, losses, input_y = sess.run([lstm.loss, lstm.accuracy, lstm.losses, lstm.input_y], feed_dict)

    perplexity = calculate_perplexities(losses, input_y, nlpdata.word_to_id[con.PAD])
    avg_perplexity = sum(perplexity) / len(perplexity)

    return loss, accuracy, perplexity



def create_graph_for_experiment(sess, experiment):

    if experiment == "A":
        return LSTM_A.LSTM_A(
            vocab_size = con.vocab_size,
            embedding_size = con.embedding_size,
            hidden_size= con.experiment_A_hidden_size,
            seq_len= con.MAX_SEQLEN
        )

    elif experiment == "B":

        return LSTM_B.LSTM_B(
            vocab_size=con.vocab_size,
            embedding_size=con.embedding_size,
            hidden_size=con.experiment_B_hidden_size,
            seq_len=con.MAX_SEQLEN
        )

    elif experiment == "C":

        return LSTM_C.LSTM_C(
            vocab_size=con.vocab_size,
            embedding_size=con.embedding_size,
            hidden_size=con.experiment_C_hidden_size,
            hidden_size2= con.experiment_C_hidden_size2,
            seq_len=con.MAX_SEQLEN
        )

    else: print("Experiment {} does not exist.".format(experiment))



for experiment_run in ["A","B","C"]:
    tf.reset_default_graph()

    with tf.Session() as sess:

        nlpdata = datasets.NLUProject1Dataset()
        lstm = create_graph_for_experiment(sess,experiment_run)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)

        gradients, variables = zip(*optimizer.compute_gradients(lstm.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        # Initialize all variables, saver and finalize the graph
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        sess.graph.finalize()

        if experiment_run == "B" or experiment_run == "C":
            pretrained_embedding = prep.load_embedding(nlpdata.word_to_id, con.embedding_path, con.embedding_size,
                                                       con.vocab_size)
            _ = sess.run(lstm.W, feed_dict={lstm.W: pretrained_embedding})

        batches = prep.batch_iter(list(zip(nlpdata.x, nlpdata.y)), con.BATCH_SIZE, con.n_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(lstm, x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % con.evaluate_every == 0:
                total_loss_eval, total_acc_eval, total_perp_eval = [], [], []
                eval_batches = prep.batch_iter(list(zip(nlpdata.eval_x, nlpdata.eval_y)),  con.BATCH_SIZE, 1)

                for eval_batch in eval_batches:
                    eval_x_batch, eval_y_batch = zip(*eval_batch)
                    eval_loss, eval_acc, eval_perplexity = dev_step(lstm, eval_x_batch, eval_y_batch)

                    total_loss_eval.append(eval_loss)
                    total_acc_eval.append(eval_acc)
                    total_perp_eval.extend(eval_perplexity)

                total_loss_eval = np.array(total_loss_eval)
                total_acc_eval = np.array(total_acc_eval)
                total_perp_eval = np.array(total_perp_eval)

                print(
                    "\n Evaluation of current model: loss {:g}, acc {:g}, perplexity {:g}".format(np.mean(total_loss_eval),
                                                                                                  np.mean(total_acc_eval),
                                                                                                  np.mean(total_perp_eval)))
                print("")

            if current_step % 30000 == 0:
                time_str = datetime.datetime.now().isoformat()
                saver.save(sess, './snapshots/Experiment_{}'.format(experiment_run), global_step=global_step)
