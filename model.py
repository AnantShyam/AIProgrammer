import torch
import gensim
import dataset
import numpy as np
import keras_nlp
from tqdm import tqdm
import tensorflow as tf


class Model(torch.nn.Module):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.sentences = data.train_dataset
        self.num_sentences, self.length_sentence = len(self.sentences), len(self.sentences[0])

        for i in range(self.num_sentences):
            for j in range(self.length_sentence):
                # check if word is an integer or not
                word = self.sentences[i][j][:]
                word = word[1:] if word[0] == '-' else word
                if word.isnumeric():
                    self.sentences[i][j] = 'Integer'

        self.max_len = max([len(sentence) for sentence in self.sentences])
        self.embeddings = gensim.models.Word2Vec(self.sentences, min_count=1, vector_size=self.max_len)

        self.word_trained_embeddings = {}
        word_vocab = list(self.embeddings.wv.key_to_index.keys())
        for word in word_vocab:
            self.word_trained_embeddings[word] = self.embeddings.wv[word]

        # add positional embeddings
        for word in self.word_trained_embeddings:
            word_embedding = self.word_trained_embeddings[word]
            # try putting word_embedding into a 2d array
            word_embedding = np.array(word_embedding)
            word_embedding_size = len(word_embedding)
            word_embedding = word_embedding.reshape((word_embedding_size, 1))

            positional_encoding = keras_nlp.layers.SinePositionEncoding()(word_embedding)
            new_word_embedding = word_embedding + positional_encoding
            new_word_embedding = torch.from_numpy(np.array(new_word_embedding))
            new_word_embedding = torch.cat(tuple([i for i in new_word_embedding]), 0)
            self.word_trained_embeddings[word] = new_word_embedding

        # for word in self.word_trained_embeddings:
        #     print(word)
        #     print(self.word_trained_embeddings[word])
        #     print("----")

        self.rnn = torch.nn.RNN(input_size=self.max_len, hidden_size=30) # random initialization of RNN
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)


    def interpret_user_input(self, test_sentence):
        converted_sentence = self.data.convert_dataset([test_sentence])[0]
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

        sentence = torch.stack(tuple([i for i in embedded_sentence]), 0)
        return sentence


    def forward(self, data):
        data = self.interpret_user_input(data)
        return self.rnn(data)

    def train_model(self):
        pass

    def test_model(self):
        pass
