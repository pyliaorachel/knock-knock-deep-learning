# Image Captioning

Image captioning based on encoder-decoder model.

- Running on [Colab](https://colab.research.google.com/drive/1Dp2F2DOZG8uALnBV_J-972s6QEGucmN2?usp=sharing) (Naive model)
- Running on [Colab](https://colab.research.google.com/drive/1oAfCxGen_zY_KlamhHKcmKGZ24w2U7Gc?usp=sharing) (With attention)
- Running on [Colab](https://colab.research.google.com/drive/1GeUnHgA_dRFfEMkZ02PqXA_M430GtJoW?usp=sharing) (With attention + pretrained models)

## Dataset

[Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k?select=Images).

## Usage

```bash
# Create & activate conda environment
$ conda env create -f env.yml
$ conda activate image-captioning

# Train & test
$ python main.py <image-folder> <caption-file> <output-encoder-file> <output-decoder-file>

# Generate captions
$ python gen.py <encoder-file> <decoder-file> <image-file> <caption-file>
```

If you want to use pretrained models:

```bash
# Download GloVe embeddings from their website, then put the embedding file you want under data/
# https://nlp.stanford.edu/projects/glove/

# Train & test
$ python main.py <image-folder> <caption-file> <output-encoder-file> <output-decoder-file> --embedding-dim <emb-dim> --use-pretrained

# Generate captions
$ python gen.py <encoder-file> <decoder-file> <image-file> <caption-file> --embedding-dim <emb-dim> --use-pretrained
```

## Training Info We use

With attention:

```bash
BATCH_SIZE: 32
EMBEDDING_DIM: 256
DEC_HIDDEN_DIM: 256
LR: 1e-05
ENCODER DROPOUT: 0.2
DECODER DROPOUT: 0.2
EPOCHS: 100
LOG_INTERVAL: 10
USE PRETRAINED: False
Training set size: 6472
Test set size: 1619
Vocab size: 4660
```

With attention and pretrained DenseNet + GloVe embeddings:

TODO

## Reference

- [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
