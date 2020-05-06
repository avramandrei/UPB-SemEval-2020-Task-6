import os
import numpy as np


def compute_results_mean_std(path, outpath):
    models_values = {}
    models_results = {}

    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            model_name = filename.split("+")[0]
            if model_name not in models_values:
                models_values[model_name] = [[], [], [], []]

            with open(os.path.join(path, filename)) as file:
                tokens = file.readline().split(",")

                for i, token in enumerate(tokens):
                    value = float(token.split(":")[1][1:])
                    models_values[model_name][i].append(value)

    print(models_values)

    for model_name, values in models_values.items():
        acc_mean = np.mean(values[0])
        acc_std = np.std(values[0])

        prec_mean = np.mean(values[1])
        prec_std = np.std(values[1])

        rec_mean = np.mean(values[2])
        rec_std = np.std(values[2])

        f1_mean = np.mean(values[3])
        f1_std = np.std(values[3])

        models_results[model_name] = [(acc_mean, acc_std), (prec_mean, prec_std), (rec_mean, rec_std), (f1_mean, f1_std)]

    print(models_results)

    with open(os.path.join(outpath, "results.txt"), "w") as file:
        for model_name, results in models_results.items():
            file.write("{} - Acc: ({}, {}), Prec: ({}, {}), Rec: ({}, {}), F1: ({}, {})\n".format(model_name,
                                                                                                results[0][0], results[0][1],
                                                                                                results[1][0], results[1][1],
                                                                                                results[2][0], results[2][1],
                                                                                                results[3][0], results[3][1],))



compute_results_mean_std("models", "results")