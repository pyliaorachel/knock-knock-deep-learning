# Chinese Text Generation Project

Text generation system based on a mixed corpus of 《毛澤東語錄》(Quotations From Chairman Mao Tse-Tung) and《論語》(Confucian Analects).

Based on Transformer. Reference: [PyTorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## Usage

```bash
# Create & activate conda environment
$ conda env create -f env.yml
$ conda activate chinese-text-generation

# Train & test with default params
$ python main.py data/corpus.txt output/model.pt

# Train with other settings, see help
$ python main.py -h

# Generate text for human evaluation
$ python gen.py data/corpus.txt output/model.pt

# Generate text with other settings, see help
$ python gen.py -h
```
