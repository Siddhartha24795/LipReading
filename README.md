# LipReading

Main repository for LipReading with Deep Neural Networks

## Introduction

The goal is to implement LipReading: Similar to how end-to-end Speech
Recognition systems work, mapping high-fidelity speech audio to sensible
characters and word level outputs, we will do the same for "speech visuals". 
In particular, we will take video frame input, extract the relevant mouth/chin
signals as input to map to characters and words.

## Overview


- [Architecture](#architecture): High level pipeline
  - [Vision Pipeline](#vision-pipeline)
  - [NLP Pipeline](#nlp-pipeline)
  - [Datasets](#datasets)
- [Setup](#setup): Quick setup and installation instructions
  - [SpaCy Setup](#spacy-setup): Setup for NLP utilities.
  - [Data Directories Structure](#data-directories-structure): How data files are organized
  - [Collecting Data](#collecting-data): See [README_DATA_COLLECTION.md](./README_DATA_COLLECTION.md)
  - [Getting Started](#getting-started): Finally get started on running things
    - [Tutorial on Configuration files](#configuration): Tutorial on how to run executables via a config file
    - [Download Data](#download-data): Collect raw data from Youtube.
    - [Generate Dataview](#generate-dataview): Generate dataview from raw data.
  - [Train Model](#train-model): :train: Train :train:
    - [Examples](#examples): Example initial configurations to experiment.
  - [Tensorboard Visualization](#tensorboard-visualization)
- [Other Resources](#other-resources): Collection of reading material, and
  projects
  
 


- [x] Download Data (926 videos)
- [x] Build Vision Pipeline (1 week) [in review](https://https://github.com/Siddhartha24795/LipReading/projects/2#card-14669202)
- [x] Build NLP Pipeline (1 week) [wip](https://https://github.com/Siddhartha24795/LipReading/projects/2#card-14669211)
- [x] Build Loss Fn and Training Pipeline (2 weeks) [wip](https://github.com/Siddhartha24795/LipReading/projects/2#card-14669251)
- [x] Train :train: and Ship :ship: [wip](https://https://github.com/Siddhartha24795/LipReading/projects/2#card-14669014)
  
## Architecture

There are two primary interconnected pipelines: a "vision" pipeline for extracting
the face and lip features from video frames, along with a "nlp-inspired"
pipeline for temporally correlating the sequential lip features into the final
output.

Here's a quick dive into tensor dimensionalities

### Vision Pipeline

```javascript
Video -> Frames       -> Face Bounding Box Detection      -> Face Landmarking    
Repr. -> (n, y, x, c) -> (n, (box=1, y_i, x_i, w_i, h_i)) -> (n, (idx=68, y, x))   
```

### NLP Pipeline

```javascript
 -> Letters  ->  Words    -> Language Model 
 -> (chars,) ->  (words,) -> (sentences,)
```

### Datasets

- `all`: 926 videos (projected, not generated yet)
- `large`: 464 videos (failed at 35/464)
- `medium`: 104 videos (currently at 37/104)
- `small`: 23 videos 
- `micro`: 6 videos
- `nano`: 1 video

## Setup

0. Clone this repository and install the requirements. We will be using `python3`.
 
Please make sure you run python scripts, setup your `PYTHONPATH` at `./`, as well as a workspace env variable.

```bash
git clone git@github.com:Siddhartha24795/LipReading.git 
# (optional, setup venv) cd LipReading; python3  -m venv .
```

1. Once the repository is cloned, the last step for setup is to setup the repository's `PYTHONPATH` and workspace environment variable to take advantage of standardized directory utilities in [`./src/utils/utility.py`](src/utils/utility.py)

Copy the following into your `~/.bashrc`

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/LipReading/" 
export LIP_READING_WS_PATH="/path/to/LipReading/"
```

2. Install the simple `requirements.txt` with `PyTorch` with CTCLoss, `SpaCy`, and others.

On MacOS for CPU capabilities only.

```bash
pip3 install -r requirements.macos.txt
```

On Ubuntu, for GPU support

```bash
pip3 install -r requirements.ubuntu.txt
```

### SpaCy Setup

We need to install a pre-built English model for some capabilities

```bash
python3 -m spacy download en
```

### Data Directories Structure

This allows us to have a simple standardized directory structure for all our datasets, raw data, model weights, logs, etc.

```text
./data/
  --/datasets (numpy dataset files for dataloaders to load)
  --/raw      (raw caption/video files extracted from online sources)
  --/weights  (model weights, both for training/checkpointing/running)
  --/tb       (Tensorboard logging)
  --/...
```

See [`./src/utils/utility.py`](src/utils/utility.py) for more.

## Getting Started

Now that the dependencies are all setup, we can finally do stuff!

### Configuration

Each of our "standard" scripts in `./src/scripts` (i.e. not `./src/scripts/misc`) take the standard `argsparse`-style 
arguments. For each of the "standard" scripts, you will be able to pass `--help` to see the expected arguments.
To maintain reproducibility, cmdline arguments can be written in a raw text file with one argument per line.

e.g. for `./config/gen_dataview/nano`

```text
--inp=StephenColbert/nano 
``` 

Represent the arguments to pass to `./src/scripts/generate_dataview.py`, automatically passable via 

```bash
./src/scripts/generate_dataview.py $(cat ./config/gen_dataview/nano)
```

The arguments will be used from left-to-right order, so if arguments are repeated, they will be overwritten by the latter settings. This allows for modularity in configuring hyperparameters.

(For demonstration purposes, not a working example)
```bash
./src/scripts/train.py \
    $(cat ./config/dataset/large) \
    $(cat ./config/train/model/small-model) \
    $(cat ./config/train/model/rnn/lstm) \
    ...
```

## Train Model

3. Train Model

```bash
./src/scripts/train.py
```

### Examples

#### Training on Micro

```bash
./src/scripts/train_model.py $(cat ./config/train/micro)
```

## Tensorboard Visualization

See [README_TENSORBOARD.md](README_TENSORBOARD.md)



