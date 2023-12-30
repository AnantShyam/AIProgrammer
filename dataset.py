import random
import string
import torch
from torch.utils.data import DataLoader

class Dataset:

    def __init__(self):
        self.train_dataset = self.build_training_dataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def build_training_dataset(self):
        train_dataset = []
        with open('data/train_data.txt', 'r') as f:
            with open('data/train_data_answers.txt', 'r') as g:
                queries, answers = [], []
                for _, query in enumerate(f.readlines()):
                    queries.append(query[:-1]) # remove newline character
                for _, answer in enumerate(g.readlines()):
                    answers.append(answer[:-1]) # remove newline character
                for i in range(len(queries)):
                    train_dataset.append(f"START {queries[i]} {answers[i]} END")
        return train_dataset




