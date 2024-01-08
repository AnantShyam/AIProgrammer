import dataset
import random
import string
import model
import argparse
import time

if __name__ == "__main__":
    # generate_training_dataset()
    parser = argparse.ArgumentParser(description="Process User Command Line Arguments")
    parser.add_argument("command")
    arguments = parser.parse_args()

    start = time.time()
    ds = dataset.Dataset()
    end = time.time()
    print(end - start)

    start = time.time()
    md = model.Model(ds)
    end = time.time()
    print(end - start)

    user_input = arguments.command

    output, _ = md.forward(user_input)
    # print(output)
    # print(output.shape)
    # print(md.interpret_user_input('Declare a variable l to be 27'))
    md.train_model()
