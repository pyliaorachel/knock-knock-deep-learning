import argparse

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from skimage.transform import resize

from model_att import ImageEncoder, CaptionDecoder 
from dataset import ImageCaptionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate caption for image')
    parser.add_argument('encoder', type=str, metavar='F',
                        help='encoder model for caption generation')
    parser.add_argument('decoder', type=str, metavar='F',
                        help='decoderfor model caption generation')
    parser.add_argument('image', type=str,
                        help='image file')
    parser.add_argument('caption_path', type=str,
                        help='path to caption data file')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='embedding dimension for characters in training model (default: 256)')
    parser.add_argument('--dec-hidden-dim', type=int, default=256,
                        help='hidden state dimension in decoder model (default: 256)')
    parser.add_argument('--max-seq-len', type=int, default=50,
                        help='max seq length for generated caption (default: 50)')
    return parser.parse_args()

def gen_caption(encoder, decoder, img, dataset, args):
    encoder.eval()
    decoder.eval()

    vocab = dataset.vocab
    word_to_int = dataset.word_to_int
    int_to_word = dataset.int_to_word
    start_token_idx = word_to_int['<start>']
    end_token_idx = word_to_int['<end>']

    # Add 1 dimension for batch
    img = img.unsqueeze(0)

    # Encode image
    img_embedding = encoder(img)

    # Decode caption
    caption_idx = decoder.decode_to_end(img_embedding, len(vocab), start_token_idx, end_token_idx)
    caption = ' '.join([int_to_word[i] for i in caption_idx[1:-1]])
    print(caption)

def main():
    args = parse_args()
    
    # Prepare dataset (for vocab) 
    dataset = ImageCaptionDataset('', args.caption_path)

    # Load image
    img = io.imread(args.image)
    img = resize(img, (512, 512))
    img = torch.from_numpy(img).float()
    img /= 255                  # normalize to range [0, 1]
    img = img.permute(2, 0, 1)  # (w, h, c) -> (c, w, h)

    # Load model
    encoder = ImageEncoder('cpu')
    decoder = CaptionDecoder('cpu', len(dataset.vocab), embedding_dim=args.embedding_dim,
                             dec_hidden_dim=args.dec_hidden_dim)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))

    # Generate caption
    gen_caption(encoder, decoder, img, dataset, args)

if __name__ == '__main__':
    main()
