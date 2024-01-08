import dataset
import random
import string
import model


def convert_dataset(data):
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


def generate_training_dataset():
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


if __name__ == "__main__":
    # generate_training_dataset()
    ds = dataset.Dataset()
    md = model.Model(ds)
    print(md.interpret_user_input('Declare a variable l to be 27'))
    md.train_model()
