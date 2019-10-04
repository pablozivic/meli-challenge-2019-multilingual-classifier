# MeLi Data Challenge 2019
Fifth place solution (balanced accuracy: 0.9108) for [MercadoLibre](https://ml-challenge.mercadolibre.com/leaderboard)'s text classification challenge.

## Model

The solution is a word level ensemble (average) of 10 LSTMs trained on FastText [MUSE](https://github.com/facebookresearch/MUSE) (Multilingual Unsupervised) embeddings.
Our objective was to allow transfer learning between languages, mapping every word on the same space.

Our model is trained with Adam in two stages:
1. We first make sure the LSTM learns the embedding space given by MUSE.
2. We then fine tune the embeddings for words in the vocabulary of MUSE, and learn the embeddings of missing words.

Overall, the model is simple. More work should be done to improve the vocabulary, train with subsampling and 
iterate the model architecture to make it faster and more expressive.

## Setting up resources

To set up the environment with all the datasets and resources, you must first call:
```
bash get_datasets.sh
bash get_embeddings.sh
```

## Installing and running package

You can install the package in development mode and get our submission.
```
pip3 install -e .
python3 -m multilingual_title_classifier.src.train
python3 -m multilingual_title_classifier.src.submission
```

## Running Docker image

You must first install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the Docker image with a GPU.

Then, install the base image we use:
```
docker pull nvidia/cuda
```

Finally, build and run our image:
```
docker build -t multilingual_title_classifier .
docker run --runtime=nvidia --ipc=host --mount source=${path_to_resources},target=/home/user/resources,type=bind multilingual_title_classifier
```