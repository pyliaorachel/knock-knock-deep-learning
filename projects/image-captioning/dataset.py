import csv
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize


class ImageCaptionDataset(Dataset):
    def __init__(self, img_root, caption_path, img_size=512, seq_len=128, should_normalize=False):
        super(ImageCaptionDataset).__init__()
        self.img_root = img_root
        self.caption_path = caption_path
        self.img_size = img_size
        self.seq_len = seq_len
        # https://pytorch.org/vision/stable/models.html
        self.should_normalize = should_normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.parse_data()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = self.parse_image(self.img_files[i])
        caption_list = self.caption_list[i]
        caption_lengths = self.caption_lengths[i]
        return img, caption_list, caption_lengths

    def parse_data(self):
        img_files, captions = [], [] 
        img_set = set()
        with open(self.caption_path, 'r') as f:
            rd = csv.reader(f, delimiter=',')
            next(rd) # ignore header
            for row in rd:
                img_file, caption = row
                # Just take one caption to be simpler
                if img_file in img_set:
                    continue
                img_set.add(img_file)

                img_files.append(img_file)
                captions.append(caption)

        caption_list, caption_lengths = self.process_captions(captions)

        self.img_files = img_files
        self.caption_list = caption_list
        self.caption_lengths = caption_lengths
        self.n = len(caption_list)

    def parse_image(self, img_file):
        """
        Load image file, convert to tensor
        """
        img_path = os.path.join(self.img_root, img_file)
        img = io.imread(img_path)
        img = resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).float()
        img /= 255                  # normalize to range [0, 1]
        img = img.permute(2, 0, 1)  # (w, h, c) -> (c, w, h)
        if self.should_normalize:
            img = self.normalize(img)
        return img

    def process_captions(self, captions):
        """
        Build vocab over all captions, convert captions to ints with start and end tokens
        """
        # Add special tokens to vocab
        vocab = set()
        vocab.add('<start>')
        vocab.add('<end>')
        vocab.add('<pad>')

        # Build vocabulary
        words = [set(caption.split(' ')) for caption in captions]
        vocab = vocab.union(set.union(*words))

        # Map word to int / int to word
        word_to_int = dict((w, i) for i, w in enumerate(vocab))
        int_to_word = dict((i, w) for i, w in enumerate(vocab))

        # Map captions to int
        caption_list = []
        caption_lengths = []
        for caption in captions:
            words = ['<start>'] + caption.split(' ') + ['<end>']
            caption_length = len(words)
            words += ['<pad>'] * (self.seq_len - len(words))
            mapped_captions = [word_to_int[word] for word in words]

            caption_tensor = torch.tensor(mapped_captions, dtype=torch.long)

            caption_list.append(caption_tensor)
            caption_lengths.append(caption_length)

        # Store vocab info
        self.word_to_int, self.int_to_word, self.vocab = word_to_int, int_to_word, vocab

        return caption_list, caption_lengths
