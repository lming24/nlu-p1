import tensorflow as tf
import datasets
import preprocessing as prep
import constants as con
import datetime
import numpy as np

data_path = "data/sentences_test"
dataset = open(data_path, 'r').readlines()
#dataset = list(filter(lambda x: len(x.split()) <= 28 - 2, dataset))
print(len(dataset))
i = 0 

for sentence in dataset:
	a = len(sentence.split())
	if a <= 28:
		i = i + 1

print(i)

#dataset = list(filter(lambda x: len(x.split()) < 28 - 2, dataset))
#print(len(dataset))