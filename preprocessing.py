from collections import Counter
import numpy as np
from gensim import models
import constants as con


def load_dataset(data_path, limit=None):
    dataset = open(data_path, 'r').readlines()

    if limit:
        dataset = dataset[:limit]
    print(dataset[0])
    # ignore sentences longer than 30 words
    dataset = list(filter(lambda x: len(x.split()) <= con.MAX_SEQLEN - 2, dataset))
    dataset = [[con.BOS] + data.split() + [con.EOS] + [con.PAD] * (con.MAX_SEQLEN - len(data.split()) - 2) for data in dataset]
    return dataset


def create_vocab(data_path, top_f = con.vocab_size -1):
    dataset = load_dataset(data_path)

    flattened_dataset = [word for data in dataset for word in data]

    vocab = dict(Counter(flattened_dataset).most_common(top_f))
    vocab[con.UNK] = 1

    return vocab


def map_id_to_word(word_to_id):
    return {v: k for k, v in word_to_id.items()}


def remove_uncommon_from_dataset(dataset, vocab):
    return [[con.UNK if vocab.get(word) == None else word for word in sentence] for sentence in dataset]


def convert_dataset_to_ids(dataset, vocab):
    word_to_id = dict()

    for i, key in enumerate(sorted(vocab, key=lambda key: vocab[key])):
        word_to_id[key] = i

    return word_to_id, [[word_to_id[word] for word in sentence] for sentence in dataset]


def convert_test_dataset_to_ids(dataset, word_to_id_vocab):
    return [[word_to_id_vocab[word] if word in word_to_id_vocab else word_to_id_vocab[con.UNK] for word in sentence] for sentence in dataset]


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]





def load_embedding(vocab, path, dim_embedding, vocab_size):

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))
    return external_embedding
