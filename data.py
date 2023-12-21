import random
import string


def generate_training_dataset():
    num_epochs = 10
    command_starts = ["Declare a variable"]
    with open('data/train_data.txt', 'w') as f:
        with open('data/train_data_answers.txt', 'w') as g:
            for _ in range(num_epochs):
                for command_start in command_starts:
                    random_letter = random.choice(string.ascii_letters)
                    random_integer = random.randint(-50, 50) # start with a small range for now
                    query = command_start + f" {random_letter} to be {random_integer}"
                    f.write(f"{query} \n")
                    g.write(f"{random_letter} = {random_integer} \n")


generate_training_dataset()
