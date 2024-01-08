import random
import string
import torch
from torch.utils.data import DataLoader
import main


class Dataset:

    def __init__(self):
        self.train_dataset = self.build_training_dataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)


    def convert_dataset(self, data):
        """
        Converts a dataset of form ['Word1, ... WordN', .... 'Word1, ... WordN']
        to [['Word1', ... 'WordN'], ... ['Word1', ... WordN']]
        :param data:
        :return: List(List(String))
        """
        assert type(data) == list
        train = []
        for example in data:
            words, word = [], ""
            for i in example:
                if i == " ":
                    words.append(word)
                    word = ""
                else:
                    word = word + i
            if word != "":
                words.append(word)
            train.append(words)
        return train


    def replace_integers(self, query):
        # replace integers with <integer>
        # We probably don't want the model to learn embeddings for each integer
        modified_query = ""
        word = ""
        for i in query:
            if i != " ":
                word = word + query
            else:
                if not word.isnumeric():
                    query = query + word
                else:
                    query = query + "INTEGER"
        if word != " ":
            if not word.isnumeric():
                modified_query = query + word
            else:
                modified_query = query + "INTEGER"
        return modified_query


    def generate_training_dataset(self):
        num_epochs = 256
        command_starts = ["Declare a variable"]

        # Save the data in text files, so we don't need to keep generating this training data again
        with open('data/train_data.txt', 'w') as f:
            with open('data/train_data_answers.txt', 'w') as g:
                for _ in range(num_epochs):
                    for command_start in command_starts:
                        random_letter = random.choice(string.ascii_letters)
                        random_integer = random.randint(-50, 50)  # start with a small range for now
                        query = command_start + f" {random_letter} to be {random_integer}"
                        f.write(f"{query}")
                        f.write("\n")
                        g.write(f"{random_letter} = {random_integer}")
                        g.write("\n")


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
                    # queries[i] = self.replace_integers(queries[i]) - seems to be slowing program down
                    train_dataset.append(f"START {queries[i]} {answers[i]} END")

        return self.convert_dataset(train_dataset)
