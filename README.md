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

To start training the deep neural network you'll need to have a collections of songs splitted into `train/val` sets. All files should be in wav format with 8KHz sampling rate with one channel. If you have a collection of songs that you want to convert you can type

```bash
python utils/convert_songs.py -src <path_to_source_dir> -dst <path_to_destination_dir>
```

to convert a collection of mp3 files into .wav files in 8KHz with 1 channel. To handle other extensions you there is an option `-ex` indicating the extension of the input files. For example, if you enter `-ex wav` then the input files are expecting to be in a wav format. To convert the files you'll need to have `ffmpeg` installed on your machine.

A well-known and publicly available dataset utilized for such MIR tasks is the <a href="https://github.com/mdeff/fma"> FMA </a> dataset. Then, you'll need to have a collection of background noises and splitted into `background_train, background_test` sets. These noises must capture all the noise that the model will face in a realistic scenario. Examples can be found on youtube from background noises from cars, vehicles, people talking, etc. To simulate the reverbaration effects, you'll need to download a collection of impulse responses again splitted into `impulse_train, impulse_val` sets. Then, place all these folders inside `deep-audio-fingerprinting/data` and create a configuration file (`training_config.json`) for training like the one below

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

## 4. Performance

Below you can a see the results of this approach on a database comprising about 26K songs. 25K come from the FMA dataset corresponding to 30 sec audio clips, while the rest are downloaded from you tube and are full length songs. The database consists about ~2M fingerprints. Below you see the results on a recorded audio corresponding to 15 songs played sequentially one after the other. You can see a model without applying low/high pass filters and a model that is trained on such degradations. The comparison is the with the dejavu open source library. 

<table>
  <caption style="font-size: 20px;">Performance (Accuracy) of the Music Recognition Systems on Microphone Experiment.</caption>
  <thead>
    <tr>
      <th>Metrics</th>
      <th>Query length (s)</th>
      <th>low</th>
      <th>mid</th>
      <th>high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Deep Audio</td>
      <td></td>
      <td><strong>85.03%</strong></td>
      <td><strong>62.65%</strong></td>
      <td><strong>41.92%</strong></td>
    </tr>
    <tr>
      <td>Deep Audio (No Filters)</td>
      <td>2</td>
      <td>67.22%</td>
      <td>49.02%</td>
      <td>19.47%</td>
    </tr>
    <tr>
      <td>Dejavu</td>
      <td></td>
      <td>12.63%</td>
      <td>6.50%</td>
      <td>10.73%</td>
    </tr>
    <tr>
      <td>Deep Audio</td>
      <td></td>
      <td><strong>92.66%</strong></td>
      <td><strong>80.06%</strong></td>
      <td><strong>62.52%</strong></td>
    </tr>
    <tr>
      <td>Deep Audio (No Filters)</td>
      <td>5</td>
      <td>80.38%</td>
      <td>65.39%</td>
      <td>31.10%</td>
    </tr>
    <tr>
      <td>Dejavu</td>
      <td></td>
      <td>59.21%</td>
      <td>40.94%</td>
      <td>47.24%</td>
    </tr>
    <tr>
      <td>Deep Audio</td>
      <td></td>
      <td><strong>94.52%</strong></td>
      <td><strong>90.32%</strong></td>
      <td><strong>74.19%</strong></td>
    </tr>
    <tr>
      <td>Deep Audio (No Filters)</td>
      <td>10</td>
      <td>84.19%</td>
      <td>78.39%</td>
      <td>41.29%</td>
    </tr>
    <tr>
      <td>Dejavu</td>
      <td></td>
      <td>83.60%</td>
      <td>70.98%</td>
      <td>71.29%</td>
    </tr>
    <tr>
      <td>Deep Audio</td>
      <td></td>
      <td><strong>97.56%</strong></td>
      <td><strong>90.24%</strong></td>
      <td>80.98%</td>
    </tr>
    <tr>
      <td>Deep Audio (No Filters)</td>
      <td>15</td>
      <td>86.34%</td>
      <td>80.98%</td>
      <td>48.29%</td>
    </tr>
    <tr>
      <td>Dejavu</td>
      <td></td>
      <td>93.33%</td>
      <td>77.14%</td>
      <td><strong>83.33%</strong></td>
    </tr>
  </tbody>
</table>

## 5. Live Demonstration

Below you can see a live demonstration of the music recognition system in a query length corresponding to 5 sec. We use "sequence search" to predict the time offset of the query also. The database consists of ~26K songs and ~2M fingerprints in this case.




https://github.com/ChrisNick92/deep-audio-fingerprinting/assets/91954890/435db9cf-b96e-4e74-b331-e759d387390a


