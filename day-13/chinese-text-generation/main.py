import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import Net


def parse_args():
    parser = argparse.ArgumentParser(description='Chinese text generation based on LSTM seq2seq model')
    parser.add_argument('corpus', type=str,
                        help='training corpus file')
    parser.add_argument('output_model', type=str,
                        help='output model file')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='input sequence length (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size (default: 32)')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='embedding dimension for characters in corpus (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='hidden state dimension (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='number of batches to wait before logging status (default: 10)')
    return parser.parse_args()

def train(model, optimizer, data, args):
    model.train()

    losses = []
    for epoch in range(args.epochs):
        total_loss = 0
        i = 0
        for batch_i, (seq_in, target) in enumerate(data):
            # Train
            optimizer.zero_grad()
            output = model(seq_in) # seq_len x batch_size x |V|
            loss = F.cross_entropy(output.view(-1, output.shape[-1]), target.t().reshape(-1))
            loss.backward()
            optimizer.step()

            # Log training status
            i += 1
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

def test(model, data, args):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for seq_in, target in data:
            output = model(seq_in)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]), target.t().reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data)
    print('Test Loss: {:.6f}'.format(avg_loss))

def main():
    args = parse_args()

    print('BATCH_SIZE: {}'.format(args.batch_size))
    print('SEQ_LENGTH: {}'.format(args.seq_length))
    print('EMBEDDING_DIM: {}'.format(args.embedding_dim))
    print('HIDDEN_DIM: {}'.format(args.hidden_dim))
    print('LR: {}'.format(args.lr))
    print('DROPOUT: {}'.format(args.dropout))
    print('EPOCHS: {}'.format(args.epochs))
    print('LOG_INTERVAL: {}'.format(args.log_interval))
    print('----------------------------')

    # Prepare data & split
    dataset = TextDataset(args.corpus, seq_length=args.seq_length)
    train_set_size = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_set_size, len(dataset) - train_set_size])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size)

    # Create model & optimizer
    model = Net(len(dataset.chars), args.embedding_dim, args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train(model, optimizer, train_dataloader, args)

    # Save model
    torch.save(model.state_dict(), args.output_model)

    # Test
    test(model, test_dataloader, args)


if __name__ == '__main__':
    main()
