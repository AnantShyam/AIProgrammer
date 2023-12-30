import torch
import gensim
import numpy as np
from tqdm import tqdm
import tensorflow as tf



class Model(torch.nn.Module):
    
    def __init__(self, dataset):
        super().__init__()
        self.sentences = dataset.train_dataset
        self.embeddings = gensim.models.Word2Vec(self.sentences, min_count=1, vector_size=100)

        self.character_trained_embeddings = {}
        word_vocab = list(self.embeddings.wv.key_to_index.keys())
        for character in word_vocab:
            self.character_trained_embeddings[character] = self.embeddings.wv[character]

        # TODO: add your own positional embeddings here
        # generate positional embeddings based on positions of character in sentence

        self.transformer = torch.nn.Transformer(
            nhead=16,
            num_encoder_layers=8,
            num_decoder_layers=8
        )

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, data):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass

    

