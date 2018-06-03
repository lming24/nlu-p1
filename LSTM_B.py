import tensorflow as tf
import constants as con

tf.reset_default_graph()

class LSTM_B(object):

    def __init__(self, vocab_size=con.vocab_size, embedding_size=con.embedding_size, hidden_size= 512, seq_len = con.MAX_SEQLEN ):

        self.input_x = tf.placeholder(tf.int32, [None, seq_len - 1], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, seq_len - 1], name="input_y")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1), name="W")
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.input_x)

        #LSTM Layer
        with tf.name_scope("lstm_layer"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, name="lstm_cell")
            initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
            state = initial_state

            outputs = []

            for t in range(seq_len - 1):
                (output, state) = lstm_cell(self.embedded_words[:, t, :], state)
                outputs.append(output)
            network_output = tf.reshape(tf.concat(outputs, 1), [-1, seq_len - 1, hidden_size])
            network_output = tf.reshape(network_output, [-1, hidden_size])

        # Output layer
        with tf.name_scope("output_layer"):
            W = tf.get_variable("W", [hidden_size, vocab_size], tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [vocab_size], tf.float32, initializer=tf.zeros_initializer())

            self.logits = tf.add(tf.matmul(network_output, W), b)
            self.logits = tf.reshape(self.logits, [-1, seq_len - 1, vocab_size])

            self.predictions = tf.argmax(self.logits, 2, name="predictions")

            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y,
                                                                         name="losses")
            self.loss = tf.reduce_mean(self.losses, name="loss")

        # Calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")