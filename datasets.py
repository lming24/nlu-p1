import preprocessing as prep
import constants as con
import pickle
from pathlib import Path

my_file = Path("dataset.pkl")
test_file = Path("test.pkl")
sentence_continuation_file = Path("sentencecontinuation.pkl")

class NLUProject1Dataset(object):

    def __init__(self):

        if my_file.is_file():
            print("Using Pickle to restore serialized objects.")

            with open('dataset.pkl', 'rb') as input:
                self.word_to_id = pickle.load(input)
                self.id_to_word = pickle.load(input)
                self.y = pickle.load(input)
                self.x = pickle.load(input)
                self.eval_y = pickle.load(input)
                self.eval_x = pickle.load(input)

        else:
            #Create datasets for training and evaluation
            dataset = prep.load_dataset(con.data_path)
            eval_dataset = prep.load_dataset(con.eval_path)

            #
            vocab = prep.create_vocab(con.data_path)
            common_dataset = prep.remove_uncommon_from_dataset(dataset, vocab)

            word_to_id, number_ver = prep.convert_dataset_to_ids(common_dataset, vocab)
            number_ver_eval = prep.convert_test_dataset_to_ids(eval_dataset, word_to_id)

            self.word_to_id = word_to_id
            self.id_to_word = prep.map_id_to_word(word_to_id)

            self.y = [sentence[1:] for sentence in number_ver]
            self.x = [sentence[:-1] for sentence in number_ver]

            self.eval_y = [sentence[1:] for sentence in number_ver_eval]
            self.eval_x = [sentence[:-1] for sentence in number_ver_eval]

            with open('dataset.pkl', 'wb') as output:

                pickle.dump(self.word_to_id , output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.id_to_word, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.y, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.x , output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.eval_y , output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.eval_x, output, pickle.HIGHEST_PROTOCOL)

class NLUProject1TestDataset():

    def __init__(self):

        if test_file.is_file():
            print("Using Pickle to restore serialized objects of test dataset.")

            with open('test.pkl', 'rb') as input:
                self.test_y = pickle.load(input)
                self.test_x = pickle.load(input)


        else:
            nlu_data = NLUProject1Dataset()

            test_dataset = prep.load_dataset(con.test_perp_path)
            number_ver_test = prep.convert_test_dataset_to_ids(test_dataset, nlu_data.word_to_id)

            self.test_y = [sentence[1:] for sentence in number_ver_test]
            self.test_x = [sentence[:-1] for sentence in number_ver_test]

            with open('test.pkl', 'wb') as output:

                pickle.dump(self.test_y, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.test_x, output, pickle.HIGHEST_PROTOCOL)

class NLUProject1SentenceContinuationDataset():

    def __init__(self):

        if sentence_continuation_file.is_file():
            print("Using Pickle to restore serialized objects of sentence continuation.")

            with open("sentencecontinuation.pkl", 'rb') as input:
                self.sc_y = pickle.load(input)
                self.sc_x = pickle.load(input)


        else:
            nlu_data = NLUProject1Dataset()

            sc_dataset = prep.load_dataset(con.test_sentence_continuation_path)
            number_ver_test = prep.convert_test_dataset_to_ids(sc_dataset, nlu_data.word_to_id)

            self.sc_y = [sentence[1:] for sentence in number_ver_test]
            self.sc_x = [sentence[:-1] for sentence in number_ver_test]

            with open("sentencecontinuation.pkl", 'wb') as output:

                pickle.dump(self.sc_y, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.sc_x, output, pickle.HIGHEST_PROTOCOL)



"""
nlu_test = NLUProject1Dataset()
for sentence in nlu_test.eval_y:
    print([ nlu_test.id_to_word[nlu_test.word_to_id[nlu_test.id_to_word[word]]] for word in sentence])
"""
"""
nlu_test = NLUProject1Dataset()
print(nlu_test.word_to_id[con.UNK])
print(nlu_test.word_to_id[con.BOS])
print(nlu_test.word_to_id[con.EOS])
print(nlu_test.word_to_id[con.PAD])


for sentence in nlu_test.eval_x:
     print([ nlu_test.id_to_word[nlu_test.word_to_id[nlu_test.id_to_word[word]]] for word in sentence])
"""