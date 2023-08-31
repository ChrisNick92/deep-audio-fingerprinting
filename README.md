# deep-audio-fingerprinting
A repository for my MSc thesis in Data Science &amp; Machine Learning @ NTUA. A deep learning approach to audio fingerprinting for recognizing songs on real time through the microphone.

This repository is a PyTorch implementation of the paper <a href="https://arxiv.org/pdf/2010.11910.pdf"> Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning </a>. In short, the goal is to train a deep neural network to extract relevant information from short audio fragments. This approach, offers an alternative way of handling the song identification problem as opposed to all Shazam-like algorithms, where they heavily relied on sophisticated feature extraction techniques.

In this repo we apply a series of data augmentations and train the neural network with contrastive loss to make it as robust as possible against to signal distortions that may arise in a realistic scenario. Below we describe:

1. How to train a deep neural network.

2. How to create and index the database.

3. How to evaluate and test the music recognition system through your microphone.

Before you do anything clone this repo by typing

```bash
git clone https://github.com/ChrisNick92/deep-audio-fingerprinting.git
```

and then install the requirements by 

```bash
pip install -r requirements.txt
```

## 1. Training the deep neural network

To start training the deep neural network you'll need to have a collections of songs splitted into `train/val` sets. A well-known and publicly available dataset utilized for such MIR tasks is the <a href="https://github.com/mdeff/fma"> FMA </a> dataset. Then, you'll need to have a collection of background noises and splitted into `background_train, background_test` sets. These noises must capture all the noise that the model will face in a realistic scenario. Examples can be found on youtube from background noises from cars, vehicles, people talking, etc. To simulate the reverbaration effects, you'll need to download a collection of impulse responses again splitted into `impulse_train, impulse_val` sets. Then, place all these folders inside `deep-audio-fingerprinting/data` and create a configuration file (`training_config.json`) for training like the one below

```
{
    "epochs": 120,
    "patience": 15,
    "batch size": 512,
    "model name": "fingerprinter",
    "optimizer": "Lamb",
    "output_path": "data/pretrained_models/",
    "background_noise_train": "data/background_train",
    "background_noise_val": "data/background_val",
    "impulse_responses_train": "data/impulse_responses",
    "impulse_responses_val": "data/impulse_responses_val",
    "train_set": ["data/train"],
    "val_set": ["data/val"],
    "loss":{
        "loss": "NTXent_Loss"
    }
}
```

Then, run

```bash
python training/trainer.py --config training/training_config.json
```

This will output `fingerprinter.pt` in `data/pretrained_models/`, containing the model's weights. Of course, you can experiment with the previous hyperparameters. 

*Note* that the quality of the model is highly dependent on the quality of the background noises & impulse responses.

## 2. Creating the database

The next step is to extract the fingerprints using the pretrained model. The features are extracted for each 1 sec audio duration with a hop length corresponding to 0.5 sec for each song to placed on the database. To extract the fingerprints create an empty folder in `deep-audio-fingerprinting/data` to place the fingerprints (e.g. `fingerprints`). Create a configuration file (`generation_config.json`) like on the one below

```
{
    "SR": 8000,
    "HOP SIZE": 4000,
    "input dirs": ["data/train", "data/val"],
    "batch size": 30,
    "weights": "data/pretrained_models/fingerprinter.pt",
    "attention": false,
    "output dir": "data/fingerprints"
}
```

Then generate the fingerprints by typing

```bash
python generation/generate_fingerprints.py --config generation/generation_config.json
```

This will output the fingerprints from the songs in the folders `train,val` in `data/fingerprints`. The fingerprints are in the form of numpy arrays, corresponding to 128-dimensional vectors. To reduce the size of the fingerprints and to achieve quick retrievals we use product quantization with a non exhaustive search based on inverted lists. We use the <a href="https://github.com/facebookresearch/faiss"> faiss </a> library to implement these techniques. The database in this case, is represented by a faiss index (`.index file`), and by a `.json` file. The index contains the quntized representations and the index contains the song names corresponding to each index on the database. To generate the index + json files, first create a configuration file like the one below

```
{
    "input_dir": "data/fingerprints",
    "output_dir": "data/faiss_indexes/",
    "name": "ivf20_pq64",
    "index": "IVF20,PQ64",
    "d": 128
}
```

The above configuration file creates a an index using 20 centroids with a product quantization with 64 subquantizers. You can experiments with the faiss indexes, the choice is dependent on the number of fingerprints to be placed on the database. In this example, the database I created consists of about ~ 2M fingerprints. You can look at the <a href="https://github.com/facebookresearch/faiss/wiki"> wiki </a> of faiss for more information about which index is appropriate depending on the database size. To create the index + json file type

```bash
python generation/create_faiss_index.py --config generation/faiss_config.json
```

This will output two files `ivf20_pq64.index, ivf20_pq64.json`, in `data/faiss_indexes/`.

## 3. Evaluate on Microphone

To test the overall performance of the system with your microphone, create a configuration file (`online_config.json`) of the form

```
{
    "SR": 8000,
    "Hop size": 4000,
    "FMB": 4000,
    "duration": 10,
    "nprobes": 5,
    "search algorithm": "sequence search",
    "weights": "data/pretrained_models/fingerprinter.pt",
    "index": "data/faiss_indexes/fingerprinter/ivf20_pq64.index",
    "json": "data/faiss_indexes/fingerprinter/ivf20_pq64.json",
    "device": "cpu",
    "neighbors": 4,
    "attention": false
}
```

The parameter "duration" corresponds to the length of the query duration in seconds. The higher the query duration, the better the results with a cost of inference time. Using the search algorithm "sequence search" we can predict the time offset of the query relative the true recording. The other option is "majority vote" which does not predict the time offset but performs slightly better. To test the performance, start playing music and type

```bash
python evaluation/online_inference.py --config evaluation/online_config.json
```

