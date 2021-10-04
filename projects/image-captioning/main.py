import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt

from dataset import ImageCaptionDataset
from model import ImageEncoder, CaptionDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Flickr8k image captioning model')
    parser.add_argument('image_folder', type=str,
                        help='folder path containing image files')
    parser.add_argument('caption_path', type=str,
                        help='path to caption data file')
    parser.add_argument('output_encoder', type=str,
                        help='output encoder model file')
    parser.add_argument('output_decoder', type=str,
                        help='output decoder model file')
    parser.add_argument('--use-curriculum-learning', action='store_true',
                        help='use curriculum learning (default: False)')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='use pretrained torchvision models (default: False)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size (default: 32)')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='embedding dimension for characters in corpus (default: 256)')
    parser.add_argument('--dec-hidden-dim', type=int, default=256,
                        help='decoder hidden state dimension (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--enc-dropout', type=float, default=0.2,
                        help='encoder dropout rate (default: 0.2)')
    parser.add_argument('--dec-dropout', type=float, default=0.2,
                        help='decoder dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='number of batches to wait before logging status (default: 10)')
    return parser.parse_args()

def print_caption(token_list, int_to_word):
    caption = ' '.join([int_to_word[token.item()] for token in token_list])
    print(caption)

def train(encoder, decoder, enc_optimizer, dec_optimizer, data, dataset, args):
    encoder.train()
    decoder.train()
    encoder.freeze_pretrained(args.use_pretrained)
    #decoder.freeze_pretrained(True)

    losses = []
    epsilon = 1 # curriculum learning
    epsilons = []
    pad_token_idx = dataset.word_to_int['<pad>']
    n_vocab = len(dataset.vocab)
    total_iters = args.epochs * len(data)
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_i, (imgs, captions, caption_lengths, _) in enumerate(data):
            # Decay teacher forcing for curriculum learning
            if args.use_curriculum_learning:
                i = epoch * len(data) + batch_i
                epsilon = 1 - 2 / (math.exp(10 * (1 - (i+1)/total_iters)) + 1)
                epsilons.append(epsilon)

            # Bring data to device
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            # Strip paddings over max caption length in batch
            max_caption_length = max(caption_lengths)
            captions = captions[:, :max_caption_length]

            # Forward pass
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            img_embeddings = encoder(imgs)
            output, captions, caption_lengths, sort_idx = decoder(img_embeddings, captions, caption_lengths, epsilon=epsilon)

            # Prepare target
            targets = captions[:, 1:] # target words are one timestep later

            # Backward pass
            loss = F.cross_entropy(output.view(-1, n_vocab), targets.reshape(-1), ignore_index=pad_token_idx)
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            # Log training status
            total_loss += loss.item()
            if batch_i % args.log_interval == 0:
                print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_i / len(data), loss.item()))
                print('Caption:')
                print_caption(captions[0], dataset.int_to_word)
                print('Target:')
                print_caption(targets[0], dataset.int_to_word)
                print('Output:')
                print_caption(output[0].squeeze(0).argmax(dim=1), dataset.int_to_word)

        losses.append(total_loss / len(data))

    if args.use_curriculum_learning:
        plt.plot(list(range(total_iters)), epsilons)
        plt.xlabel('Iterations')
        plt.ylabel('Epsilon')
        plt.title('Teacher forcing decay over epochs')
        plt.savefig('curriculum_learning.png')
        plt.clf()

    # Plot
    plt.plot(list(range(args.epochs)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Training loss over epochs')
    plt.savefig('training_loss.png')

def test(encoder, decoder, data, dataset, args):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_loss_no_tf = 0
    pad_token_idx = dataset.word_to_int['<pad>']
    n_vocab = len(dataset.vocab)
    with torch.no_grad():
        for imgs, captions, caption_lengths, img_files in data:
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            max_caption_length = max(caption_lengths)
            captions = captions[:, :max_caption_length]

            img_embeddings = encoder(imgs)
            # Teacher forcing
            output, _, _, sort_idx = decoder(img_embeddings, captions, caption_lengths)
            # No teacher forcing
            output_no_tf, _, _, sort_idx = decoder(img_embeddings, captions, caption_lengths, epsilon=0)

            captions = captions[sort_idx]
            caption_lengths = caption_lengths[sort_idx]

            targets = captions[:, 1:]

            loss = F.cross_entropy(output.view(-1, n_vocab), targets.reshape(-1), ignore_index=pad_token_idx)
            loss_no_tf = F.cross_entropy(output_no_tf.view(-1, n_vocab), targets.reshape(-1), ignore_index=pad_token_idx)

            total_loss += loss.item()
            total_loss_no_tf += loss_no_tf.item()

    avg_loss = total_loss / len(data)
    avg_loss_no_tf = total_loss_no_tf / len(data)
    print('Test Loss: {:.6f}'.format(avg_loss))
    print('Test Loss (no teacher forcing): {:.6f}'.format(avg_loss_no_tf))

def main():
    args = parse_args()

    print('BATCH_SIZE: {}'.format(args.batch_size))
    print('EMBEDDING_DIM: {}'.format(args.embedding_dim))
    print('DEC_HIDDEN_DIM: {}'.format(args.dec_hidden_dim))
    print('LR: {}'.format(args.lr))
    print('ENCODER DROPOUT: {}'.format(args.enc_dropout))
    print('DECODER DROPOUT: {}'.format(args.dec_dropout))
    print('EPOCHS: {}'.format(args.epochs))
    print('LOG_INTERVAL: {}'.format(args.log_interval))
    print('USE PRETRAINED: {}'.format(args.use_pretrained))
    print('USE CURRICULUM LEARNING: {}'.format(args.use_curriculum_learning))

    # Prepare data & split
    dataset = ImageCaptionDataset(args.image_folder, args.caption_path)
    train_set, test_set = dataset.random_split(train_portion=0.8)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
    print('Training set size: {}'.format(len(train_set)))
    print('Test set size: {}'.format(len(test_set)))
    print('Vocab size: {}'.format(len(dataset.vocab)))
    print('----------------------------')

    # Create model & optimizer
    encoder = ImageEncoder(device, pretrained=args.use_pretrained).to(device)
    decoder = CaptionDecoder(device, len(dataset.vocab), embedding_dim=args.embedding_dim,
                             enc_hidden_dim=encoder.hidden_dim, dec_hidden_dim=args.dec_hidden_dim, dropout=args.dec_dropout,
                             use_pretrained_emb=args.use_pretrained, word_to_int=dataset.word_to_int).to(device)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # Train
    train(encoder, decoder, enc_optimizer, dec_optimizer, train_dataloader, dataset, args)

    # Save model
    torch.save(encoder.cpu().state_dict(), args.output_encoder)
    torch.save(decoder.cpu().state_dict(), args.output_decoder)
    encoder.to(device)
    decoder.to(device)

    # Test
    test(encoder, decoder, test_dataloader, dataset, args)


if __name__ == '__main__':
    main()
