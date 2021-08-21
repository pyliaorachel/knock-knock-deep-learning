import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt

from dataset import ImageCaptionDataset
from model_att import ImageEncoder, ImageEncoderPretrained, CaptionDecoder

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

def train(encoder, decoder, enc_optimizer, dec_optimizer, data, args):
    encoder.train()
    decoder.train()

    losses = []
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_i, (imgs, captions, caption_lengths) in enumerate(data):
            # Bring data to device
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            # Forward pass
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            img_embeddings = encoder(imgs)
            output, captions, caption_lengths, sort_idx = decoder(img_embeddings, captions, caption_lengths)

            # Prepare target
            targets = captions[:, 1:] # target words are one timestep later

            # Trick to ignore paddings
            output = pack_padded_sequence(output, caption_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, caption_lengths, batch_first=True).data

            # Backward pass
            loss = F.cross_entropy(output, targets)
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            # Log training status
            total_loss += loss.item()
            if batch_i % args.log_interval == 0:
                print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_i / len(data), loss.item()))
        losses.append(total_loss / len(data))

    # Plot
    plt.plot(list(range(args.epochs)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Training loss over epochs')
    plt.savefig('training_loss.png')

def test(encoder, decoder, data, args):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    with torch.no_grad():
        for imgs, captions, caption_lengths in data:
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            img_embeddings = encoder(imgs)
            output, captions, caption_lengths, sort_idx = decoder(img_embeddings, captions, caption_lengths)

            targets = captions[:, 1:]
            output = pack_padded_sequence(output, caption_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, caption_lengths, batch_first=True).data

            loss = F.cross_entropy(output, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data)
    print('Test Loss: {:.6f}'.format(avg_loss))

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

    # Prepare data & split
    dataset = ImageCaptionDataset(args.image_folder, args.caption_path, should_normalize=args.use_pretrained)
    train_set_size = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_set_size, len(dataset) - train_set_size])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
    print('Training set size: {}'.format(len(train_set)))
    print('Test set size: {}'.format(len(test_set)))
    print('Vocab size: {}'.format(len(dataset.vocab)))
    print('----------------------------')

    # Create model & optimizer
    if args.use_pretrained:
        encoder = ImageEncoderPretrained(device).to(device)
        enc_hidden_dim = 2208
    else:
        encoder = ImageEncoder(device, dropout=args.enc_dropout).to(device)
        enc_hidden_dim = 1024
    decoder = CaptionDecoder(device, len(dataset.vocab), embedding_dim=args.embedding_dim,
                             enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=args.dec_hidden_dim, dropout=args.dec_dropout).to(device)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # Train
    train(encoder, decoder, enc_optimizer, dec_optimizer, train_dataloader, args)

    # Save model
    torch.save(encoder.cpu().state_dict(), args.output_encoder)
    torch.save(decoder.cpu().state_dict(), args.output_decoder)
    encoder.to(device)
    decoder.to(device)

    # Test
    test(encoder, decoder, test_dataloader, args)


if __name__ == '__main__':
    main()
