import os
import argparse
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    args = parser.parse_args()

    entity_counter = Counter()

    for filename in os.listdir(args.data_path):
        with open(os.path.join(args.data_path, filename), encoding="utf-8", mode="r") as file:

            for line in file:
                if line is not "\n":
                    tokens = line.split("\t")
                    entity = "O" if tokens[4] == " O" else tokens[4].replace(" ", "")[2:]
                    print(tokens[4], entity)

                    entity_counter[entity] += 1

    print(entity_counter)