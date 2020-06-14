import torch
import argparse
import time
import os
import torch.nn as nn
from tqdm import tqdm
from model import NLINetwork
from preprocessing import SNLIDataset

parser = argparse.ArgumentParser(description='Parameters for training')
parser.add_argument('--hidden_size', type=int, default=300, help='size of hidden state in LSTM')
parser.add_argument('--dropout', type=float, default=0.5, help='drop out ratio in the model')
parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='size of each training batch')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--embedding_dim', type=int, default=50)
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--checkpoint', type=str, default=False)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--max_grad_norm', type=float, default=10.0)

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
torch.cuda.set_device(0)

def correct_predictions(probs, targets):
    _, pred_class = probs.max(dim=1)
    correct = (pred_class==targets).sum()
    return correct.item()

def train_epoch(model, train_iter, train_size, optimizer, criterion, epoch, max_grad_norm):
    model.train()
    device = model.device

    model.to(device)

    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    epoch_start = time.time()
    # pack iterator with tqdm wrapper
    tqdm_iter = tqdm(train_iter)

    # epoch loop
    for batch_index, data_pack in enumerate(tqdm_iter):
        batch_start = time.time()

        # raw premise, hypothesis data (with corresponding sequence lengths)
        prem, hypo, labels = data_pack.premise, data_pack.hypothesis, data_pack.label
        # print(prem[0].device)

        optimizer.zero_grad()

        logits, probs = model(prem, hypo)

        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"    \
            .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_iter.set_description(description)
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss/ len(train_iter)
    epoch_accuracy = correct_preds / train_size
    return epoch_time, epoch_loss, epoch_accuracy

def validate(model, dev_iter, dev_size, criterion):
    model.eval()
    device = model.device
    epoch_start = time.time()

    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []

    with torch.no_grad():
        for data_pack in dev_iter:
            batch_start = time.time()

            prem, hypo, labels = data_pack.premise, data_pack.hypothesis, data_pack.label

            logits, probs = model(prem, hypo)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
            all_labels.extend(labels)
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dev_iter)
    epoch_accuracy = running_accuracy / dev_size
    return epoch_time, epoch_loss, epoch_accuracy

def main():
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Load data
    snli = SNLIDataset(args.embedding_dim, args.batch_size, device=DEVICE)
    train_iter, dev_iter, test_iter = snli.train_iter, snli.dev_iter, snli.test_iter
    train_size, dev_size, test_size = snli.train_size, snli.dev_size, snli.test_size

    print("\t* Building model...")
    model = NLINetwork(num_embeddings=snli.vocab_size,
                    embedding_dim=args.embedding_dim,
                    hidden_size=args.hidden_size,
                    num_class=snli.num_class,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    vectors=snli.pretrained_vectors,
                    device=DEVICE)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1

    # plot loss curves
    epochs_count = []
    train_losses = []
    valid_losses = []

    # load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    patience_counter = 0
    for epoch in range(start_epoch, args.epochs+1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))

        epoch_time, epoch_loss, epoch_accuracy = train_epoch(model, train_iter, train_size,   \
                                            optimizer, criterion, epoch, args.max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"     \
                .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        print("* Validation for epoch{}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model, dev_iter, dev_size, criterion)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%"    \
                .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        
        # Update the optimizer's learning rate with the scheduler
        scheduler.step(epoch_accuracy)

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'best_score': best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(args.log_dir, "best.pth.tar"))
            # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                    os.path.join(args.log_dir, "slni_{}.pth.tar.format(epoch)"))
        if patience_counter > args.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

if __name__ == "__main__":
    main()
