import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset import SentimentDataset
from model import BertSentiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='BERT sentiment analysis')
    parser.add_argument('corpus', type=str,
                        help='training corpus file')
    parser.add_argument('output_model', type=str,
                        help='output model file')
    parser.add_argument('--seq-length', type=int, default=512,
                        help='input sequence length (default: 512)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='hidden dimension (default: 256)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size (default: 32)')
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
    #model.freeze_bert(True) # don't fine-tune bert to speedup model training

    n_batch = len(data)
    n_data = len(data.dataset)
    losses = []
    accs = []
    n_iters = 0
    for epoch in range(args.epochs):
        total_loss = 0
        total_correct = 0
        for batch_i, (input_ids, attention_mask, token_type_ids, target) in enumerate(data):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            target = target.to(device)

            # Train
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Log training status
            n_iters += 1
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item() # how many predictions in this batch are correct
            total_correct += correct
            if batch_i % args.log_interval == 0:
                print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}\tAccuracy: {}/{} ({:.0f}%)'.format(
                    epoch, 100. * batch_i / n_batch, loss.item(),
                    correct, len(target), 100. * correct / len(target)))
        losses.append(total_loss / len(data))
        accs.append(total_correct / n_data)

    # Plot
    plt.plot(list(range(args.epochs)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Training loss over epochs')
    plt.savefig('training_loss.png')
    
    plt.clf()
    plt.plot(list(range(args.epochs)), accs)
    plt.xlabel('Epochs')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy over epochs')
    plt.savefig('training_acc.png')

def test(model, data, args):
    model.eval()

    n_batch = len(data)
    n_data = len(data.dataset)
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, target in data:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            target = target.to(device)

            output = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # how many predictions in this batch are correct

    avg_loss = total_loss / n_batch
    print('Test Loss: {:.6f}'.format(avg_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, n_data,
        100. * correct / n_data))

def main():
    args = parse_args()

    N_CLASS = 5 # 5 classes for this sentiment dataset

    print('BATCH_SIZE: {}'.format(args.batch_size))
    print('SEQ_LENGTH: {}'.format(args.seq_length))
    print('HIDDEN_DIM: {}'.format(args.hidden_dim))
    print('LR: {}'.format(args.lr))
    print('DROPOUT: {}'.format(args.dropout))
    print('EPOCHS: {}'.format(args.epochs))
    print('LOG_INTERVAL: {}'.format(args.log_interval))

    # Prepare data & split
    dataset = SentimentDataset(args.corpus, seq_length=args.seq_length)
    train_set_size = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_set_size, len(dataset) - train_set_size])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
    print('Training set size: {}'.format(len(train_set)))
    print('Test set size: {}'.format(len(test_set)))
    print('----------------------------')

    # Create model & optimizer
    model = BertSentiment(n_class=N_CLASS, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train(model, optimizer, train_dataloader, args)

    # Save model
    torch.save(model.state_dict(), args.output_model)

    # Test
    test(model, test_dataloader, args)


if __name__ == '__main__':
    main()