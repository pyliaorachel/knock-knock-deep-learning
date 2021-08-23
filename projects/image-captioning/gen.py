import argparse

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from skimage.transform import resize

from model import ImageEncoder, ImageEncoderPretrained, CaptionDecoder 
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
    parser.add_argument('--use-pretrained', action='store_true',
                        help='use pretrained torchvision models (default: False)')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='embedding dimension for characters in training model (default: 256)')
    parser.add_argument('--dec-hidden-dim', type=int, default=256,
                        help='hidden state dimension in decoder model (default: 256)')
    parser.add_argument('--max-seq-len', type=int, default=50,
                        help='max seq length for generated caption (default: 50)')
    parser.add_argument('--k', type=int, default=3,
                        help='size for beam search (default: 3)')
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
    caption_idx = decoder.decode_to_end(img_embedding, len(vocab), start_token_idx, end_token_idx, k=args.k)
    caption = ' '.join([int_to_word[i] for i in caption_idx[1:-1]])
    print(caption)

def main():
    args = parse_args()
    
    # Prepare dataset (for vocab) 
    dataset = ImageCaptionDataset('', args.caption_path, use_pretrained=args.use_pretrained)

    # Load image
    img = dataset.parse_image(args.image)

    # Load model
    if args.use_pretrained:
        encoder = ImageEncoderPretrained('cpu')
        enc_hidden_dim = 2048
    else:
        encoder = ImageEncoder('cpu')
        enc_hidden_dim = 1024

    # No need to pass use_pretrained to decoder, as the weights will be loaded
    decoder = CaptionDecoder('cpu', len(dataset.vocab), embedding_dim=args.embedding_dim,
                             enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=args.dec_hidden_dim)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))

    # Generate caption
    gen_caption(encoder, decoder, img, dataset, args)

if __name__ == '__main__':
    main()
