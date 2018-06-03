data_path = "./data/sentences.train"
eval_path = "./data/sentences.eval"
test_sentence_continuation_path = "./data/sentences.continuation"
test_perp_path = "./data/sentences_test"

embedding_path =  "./data/wordembeddings-dim100.word2vec"

EOS = "<eos>"
BOS = "<bos>"
PAD = "<pad>"
UNK = "<unk>"

MAX_SEQLEN = 30
BATCH_SIZE = 1
n_epochs = 1
vocab_size = 20000
embedding_size = 100

#Validation Set
evaluate_every = 2000

#Experiment A
experiment_A_hidden_size = 512
experiment_B_hidden_size = 512
experiment_C_hidden_size = 1024
experiment_C_hidden_size2 = 512
