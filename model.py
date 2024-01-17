import torch
import gensim
import dataset
import numpy as np
import keras_nlp
from tqdm import tqdm
from torch.utils.data import DataLoader
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
                word = word[1:] if word[0] == '-' else word  # handle negative integers
                if word.isnumeric():
                    self.sentences[i][j] = 'Integer'

        print(self.sentences)

        self.embedding_size = 30  # can change this if needed
        self.embeddings = gensim.models.Word2Vec(self.sentences, min_count=1, vector_size=self.embedding_size)

        self.word_trained_embeddings = {}
        self.word_vocab = list(self.embeddings.wv.key_to_index.keys())

        for word in self.word_vocab:
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

        # print(self.word_trained_embeddings)

        self.num_layers = 10  # can change this if needed
        self.hidden_size = len(self.word_vocab) # can change this if needed

        self.train_data, self.initial_hidden_state = self.prepare_rnn_model_inputs()
        self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)

        self.rnn = torch.nn.RNN(input_size=30, hidden_size=self.hidden_size) # random initialization of RNN



        self.loss_function = torch.nn.MSELoss()

        self.activation = torch.nn.Softmax()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

        self.train_model()

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
                embedding = torch.rand(self.max_len) # should this self.word_embedding_size here?
            embedding = torch.from_numpy(embedding)
            embedded_sentence.append(embedding)

        sentence = torch.stack(tuple([i for i in embedded_sentence]), 0)
        return sentence


    def prepare_rnn_model_inputs(self):
        # prepare the appropriate tensors to use to train the rnn
        train_data_embeddings = []
        for sentence in self.sentences:
            sentence_embedding = []
            for word in sentence:
                embedding = self.word_trained_embeddings[word]
                sentence_embedding.append(embedding)
            sentence_embedding = np.array(sentence_embedding)
            train_data_embeddings.append(sentence_embedding)
        train_data_embeddings = torch.from_numpy(np.array(train_data_embeddings))
        h_0 = torch.rand((self.num_layers, self.hidden_size))
        return train_data_embeddings, h_0


    def forward(self, data):
        output, _ = self.rnn(data)
        return self.activation(output)


    def build_MLP(self, input_size):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, len(self.word_vocab)),
            torch.nn.Softmax()
        )
        return model


    def train_model(self):

        num_epochs = 1
        for _ in tqdm(range(num_epochs)):
            self.train()
            num_sentences = len(self.train_data) # not doing minibatching for right now
            num_sentences = 1
            for i in range(num_sentences):
                sentence = self.train_data[i]
                n_sentences, sentence_length = sentence.shape

                # reshape into 1d tensor
                concatenated_sentence = torch.reshape(sentence, (n_sentences * sentence_length, ))
                # model_output = self.forward(concatenated_sentence)

                # prediction = torch.argmax(self.forward(sentence)).item()

                self.optimizer.zero_grad()
                # insert code for doing backpropagation on the loss
                self.optimizer.step()
        pass

