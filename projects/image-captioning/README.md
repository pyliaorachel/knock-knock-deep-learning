# Image Captioning

Image captioning based on encoder-decoder model.

Running on [Colab](https://colab.research.google.com/drive/1Dp2F2DOZG8uALnBV_J-972s6QEGucmN2?usp=sharing)

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

## Reference

- [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
