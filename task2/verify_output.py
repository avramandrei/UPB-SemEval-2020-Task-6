import os


for filename in os.listdir("data/dev"):
    with open("data/dev/" + filename) as gold_file, open("output/task_2_" + filename) as pred_file:
        gold_count = 0

        for line in gold_file:
            gold_count += 1

        pred_count = 0
        for line in pred_file:
            pred_count += 1

        print(filename, gold_count, pred_count)

        assert gold_count == pred_count