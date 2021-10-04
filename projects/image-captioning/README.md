# Image Captioning

Image captioning based on encoder-decoder + attention model.

- Base model on [Colab](https://colab.research.google.com/drive/1aFnnkRAHTLmAjj7YjIisF6XmVl5Ohxko?usp=sharing)
- Pretrained model on [Colab](https://colab.research.google.com/drive/1yA_IaIxGhbU7iBrIGiS0PpQ8U9F4iSI-?usp=sharing)
- Pretrained model with curriculum learning on [Colab](https://colab.research.google.com/drive/10DjYB8wnnWjcjUj8Vo1TSKmny11IdrXL?usp=sharing)

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

If you want to enable curriculum learning:

```bash
# Download GloVe embeddings from their website, then put the embedding file you want under data/
# https://nlp.stanford.edu/projects/glove/

# Train & test
$ python main.py <image-folder> <caption-file> <output-encoder-file> <output-decoder-file> --use-curriculum-learning

# Generate captions
$ python gen.py <encoder-file> <decoder-file> <image-file> <caption-file> --use-curriculum-learning
```

## Training Info We use

Base:

```bash
BATCH_SIZE: 64
EMBEDDING_DIM: 300
DEC_HIDDEN_DIM: 512
LR: 3e-04
ENCODER DROPOUT: 0.2
DECODER DROPOUT: 0.2
EPOCHS: 50
LOG_INTERVAL: 10
USE PRETRAINED: False
USE CURRICULUM LEARNING: False
Training set size: 32360
Test set size: 8095
Vocab size: 8922
```

With pretrained:

```bash
BATCH_SIZE: 256
EMBEDDING_DIM: 300
DEC_HIDDEN_DIM: 512
LR: 0.0003
ENCODER DROPOUT: 0.2
DECODER DROPOUT: 0.2
EPOCHS: 50
LOG_INTERVAL: 10
USE PRETRAINED: True
USE CURRICULUM LEARNING: False
Training set size: 32360
Test set size: 8095
Vocab size: 8922
```

With pretrained + curriculum learning:

```bash
BATCH_SIZE: 256
EMBEDDING_DIM: 300
DEC_HIDDEN_DIM: 512
LR: 0.0003
ENCODER DROPOUT: 0.2
DECODER DROPOUT: 0.2
EPOCHS: 50
LOG_INTERVAL: 10
USE PRETRAINED: True
USE CURRICULUM LEARNING: True
Training set size: 32360
Test set size: 8095
Vocab size: 8922
```

## Reference

- [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
