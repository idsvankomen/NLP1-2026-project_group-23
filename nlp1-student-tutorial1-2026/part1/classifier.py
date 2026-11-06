import sys
from datasets import load_dataset
import experiment
import clslib
import analysis
from encoders import TokenIndicator, BagOfWords, TF_IDF, FrequencyTokenizer
from tokenizers import Tokenizer
import numpy as np
from collections import OrderedDict


def cross_validation(dataset, model, encoding_method, encoder, outcome_space,
                     max_iter):

    inverted_outcome_space = {v:k for k,v in outcome_space.items()}

    my_cross_validation_set = experiment.cross_validation_split(
            # we pair the texts with their labels
            pairs=list(zip(dataset["train"]["content"],
                           [inverted_outcome_space[x] for x in
                            dataset["train"]["emotion"]])),
            k=5,
            seed=0
            )

    my_classifiers = OrderedDict()
    my_classifiers[f'{model}_{encoding_method}-enc'] = []
    run_folder = f"ISEAR_results/{model}_{encoding_method}-enc/seed-0"

    if model == "logistic":
        my_classifiers[f'logistic_{encoding_method}-enc'].append(
            experiment.cross_validation_experiment(
                data_splits=my_cross_validation_set,
                outcome_space=outcome_space,
                encoder=encoder,
                model_option='logistic',
                save_to=run_folder,
                loglin_args={
                    'max_iter': max_iter
                }
            )
        )

    elif model == "mlp":
        my_classifiers[f'mlp_{encoding_method}-enc'].append(
            experiment.cross_validation_experiment(
                data_splits=my_cross_validation_set,
                outcome_space=outcome_space,
                encoder=encoder,
                model_option="mlp",
                save_to=run_folder,
                smlp_args={
                    'hidden_dim': 128,
                    'dropout': 0.1,
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-4,
                    'num_epochs': 10,
            }
        )
    )

    return my_classifiers


def run_experiment(model, encoding_method, tokenizer_method):

    if model not in ("logistic", "mlp"):
        print("Training method not implemented (choose logistic/mlp)")
        return
    if encoding_method not in ("Indicator", "BoW", "TF-IDF"):
        print("Encoding method not implemented (choose Indicator/BoW/TF-IDF)")
        return
    if tokenizer_method not in ("BPE", "MostFrequentWords"):
        print("Tokenizer method not implemented (choose BPE/MostFrequentWords)")
        return
    if encoding_method == "TF-IDF" and tokenizer_method != "MostFrequentWords":
        print("Warning: tokenizer method not implemented for TF-IDF. MostFrequentWords is used instead")

    dataset = load_dataset("gsri-18/ISEAR-dataset-complete")
    outcome_space = clslib.NamedClasses(["joy", "fear", "anger", "sadness",
                                         "disgust", "shame", "guilt"])


    if encoding_method == "TF-IDF" or tokenizer_method == "MostFrequentWords":
        #vs, max_iter = [int(x) for x in sys.argv[4:]]
        try:
            tokenizer = FrequencyTokenizer.load_from_file(
                "results/Frequency_tokenizer_emotions.json")
        except:
            tokenizer = FrequencyTokenizer(1000)
            tokenizer.train(dataset["train"]["content"])
            tokenizer.save("results/frequency_tokenizer_emotions.json")

    elif tokenizer_method == "BPE":
        #vs, mf, max_iter = [int(x) for x in sys.argv[4:]]
        try:
            tokenizer = Tokenizer.from_file(
            f"results/BPE_tokenizer_emotions_vs_1000_mf_2.json")
        except:
            corpus = dataset["train"]["content"]
            tokenizer = experiment.train_tokenizer(corpus, 1000, 2)
            tokenizer.save(f"results/BPE_tokenizer_emotions_vs_1000_mf_2.json")

    if encoding_method == "Indicator":
        encoder = TokenIndicator(tokenizer)
    if encoding_method == "BoW":
        encoder = BagOfWords(tokenizer)
    if encoding_method == "TF-IDF":
        encoder = TF_IDF(tokenizer)

    my_classifiers = cross_validation(dataset, model, encoding_method, encoder,
                                      outcome_space, 100)

    df = analysis.make_data_frame(my_classifiers)
    return df

def encoder_tf_idf():
    dataset = load_dataset("gsri-18/ISEAR-dataset-complete")

    try:
        tokenizer = FrequencyTokenizer.load_from_file(
            "results/Frequency_tokenizer_emotions.json")
    except:
        tokenizer = FrequencyTokenizer(1000)
        tokenizer.train(dataset["train"]["content"])
        tokenizer.save("results/frequency_tokenizer_emotions.json")

    return TF_IDF(tokenizer)

def encoder_indicator():
    dataset = load_dataset("gsri-18/ISEAR-dataset-complete")

    try:
        tokenizer = Tokenizer.from_file(
        f"results/BPE_tokenizer_emotions_vs_1000_mf_2.json")
    except:
        corpus = dataset["train"]["content"]
        tokenizer = experiment.train_tokenizer(corpus, 1000, 2)
        tokenizer.save(f"results/BPE_tokenizer_emotions_vs_1000_mf_2.json")

    return TokenIndicator(tokenizer)

if __name__ == "__main__":
    model, encoding_method, tokenizer_method = sys.argv[1:4]
    df = run_experiment(model, encoding_method, tokenizer_method)
    print(df)
    analysis.plot_box_and_points(df)