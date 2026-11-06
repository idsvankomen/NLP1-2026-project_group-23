from datasets import load_dataset
import experiment

dataset = load_dataset("gsri-18/ISEAR-dataset-complete")
corpus = dataset["train"]["content"]

vocab_size = 1000
min_frequency = 2
tokenizer = experiment.train_tokenizer(corpus, vocab_size, min_frequency)

tokenizer.save(f"results/BPE_tokenizer_emotions_vs_{vocab_size}_mf_{min_frequency}.json")