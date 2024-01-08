import torch
import gensim
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import main


class Model(torch.nn.Module):

    def __init__(self, dataset):
        super().__init__()
        self.sentences = dataset.train_dataset
        self.max_len = max([len(sentence) for sentence in self.sentences])
        self.embeddings = gensim.models.Word2Vec(self.sentences, min_count=1, vector_size=100)

        self.word_trained_embeddings = {}
        word_vocab = list(self.embeddings.wv.key_to_index.keys())
        for word in word_vocab:
            self.word_trained_embeddings[word] = self.embeddings.wv[word]

        self.model = torch.nn.RNN(20, 20) # random initialization of RNN
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)


    def interpret_user_input(self, test_sentence):
        converted_sentence = main.convert_dataset([test_sentence])[0]
        n = len(converted_sentence)

        embedded_sentence = []
        for i in range(n):
            word = converted_sentence[i]
            embedding = None
            if word in self.word_trained_embeddings:
                embedding = self.word_trained_embeddings[word]
            else:
                embedding = torch.rand(self.max_len)
            embedding = torch.from_numpy(embedding)
            embedded_sentence.append(embedding)

        sentence = torch.cat(tuple([i for i in embedded_sentence]), 0)
        return sentence



    def forward(self, data):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass
