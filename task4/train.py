import torch
import argparse
import time
import pdb
import os
import torch.nn as nn
from tqdm import tqdm
from model import BLCC
from utils import CONLL2003

parser = argparse.ArgumentParser(description='Parameters for training')
parser.add_argument('--tag_type',type=str, default='pos', help='The type of tags to be used in the model')
parser.add_argument('--hidden_size', type=int, default=100, help='size of hidden state in LSTM')
parser.add_argument('--dropout', type=float, default=0.5, help='drop out ratio in the model')
parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='size of each training batch')
parser.add_argument('--lr', type=float, default=1.5e-2, help='inital learning rate')
parser.add_argument('--embedding_dim_word', type=int, default=100)
parser.add_argument('--embedding_dim_char', type=int, default=30)
parser.add_argument('--log_root', type=str, default='./log', help='root path to place the log files')
parser.add_argument('--data_root', type=str, default='./conll2003_polished',help='root path to place the dataset')
parser.add_argument('--checkpoint', type=str, default=False)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--gamma', type=float,default=0.05, help='The coefficient for exponetial decay')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='The maximum norm for gradient clipping')
parser.add_argument('--use_charemb', type=bool, default=False, help='Whether to use char embedding in the model.')

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
torch.cuda.set_device(0)

def correct_predictions(preds, labels):
    correct = 0
    labels = labels.to('cpu')
    for pred, label in zip(preds, labels):
        length = len(pred)
        correct = correct + ((torch.Tensor(pred)+2)==label[:length]).sum().item()
    return correct

def num_batch_labels(labels):
    valid_labels = (labels != 1)
    return valid_labels.sum().item()

def train_epoch(model, train_iter, train_size, optimizer, epoch, max_grad_norm):
    model.train()
    device = DEVICE

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
        word_seq, char_seq, tags = data_pack.inputs_word, data_pack.inputs_char, data_pack.labels

        optimizer.zero_grad()

        logprob_batch = model(word_seq, char_seq, tags)
        loss = -logprob_batch   # We want to maxmize likelihood, which equals to minimize minus one
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        # correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"    \
            .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_iter.set_description(description)
    
    epoch_time = time.time() - epoch_start
    epoch_neg_likelihood = running_loss/ len(train_iter)
    # epoch_accuracy = correct_preds / train_size
    return epoch_time, epoch_neg_likelihood

def validate(model, dev_iter):
    model.eval()
    device = model.device
    epoch_start = time.time()

    running_accuracy = 0.0
    total_length = 0
    all_prob = []
    all_labels = []

    with torch.no_grad():
        for data_pack in dev_iter:
            batch_start = time.time()

            word_seq, char_seq, labels = data_pack.inputs_word, data_pack.inputs_char, data_pack.labels
            pred_labels = model(word_seq, char_seq, labels)

            running_accuracy += correct_predictions(pred_labels, labels)
            total_length += num_batch_labels(labels)
    
    epoch_time = time.time() - epoch_start
    epoch_accuracy = running_accuracy / total_length
    return epoch_time, epoch_accuracy

def test(model, test_iter):
    # test the best model
    torch.load(os.path.join(args.log_root, "best.pth.tar"))
    epoch_time, epoch_accuracy = validate(model, test_iter)
    return epoch_time, epoch_accuracy

def main():
    log_root = args.log_root
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    
    # Load data
    conll = CONLL2003(
                        batch_size=args.batch_size,
                        word_embedding_dim=args.embedding_dim_word,
                        path=args.data_root,
                        tag_type=args.tag_type,
                        device=DEVICE
                    )
    train_iter, dev_iter, test_iter = conll.train_iter, conll.val_iter, conll.test_iter
    train_size, dev_size, test_size = conll.train_size, conll.val_size, conll.test_size

    print("\t* Building model...")
    model = BLCC(
                    num_word_embeddings=conll.word_vocab_size,
                    num_char_embeddings=conll.char_vocab_size,
                    num_class=conll.num_class,
                    embedding_dim_word = args.embedding_dim_word,
                    embedding_dim_char=args.embedding_dim_char,
                    hidden_size=args.hidden_size,
                    p=args.dropout,
                    word_vectors=conll.word_vectors,
                    device=DEVICE
                )
    
    # Set up training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
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

        epoch_time, epoch_neg_likelihood = train_epoch(model, train_iter, train_size,   \
                                            optimizer, epoch, args.max_grad_norm)

        train_losses.append(epoch_neg_likelihood)
        print("-> Training time: {:.4f}s, epoch_neg_likelihood: {:.4f}"     \
                .format(epoch_time, epoch_neg_likelihood))
        
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_accuracy = validate(model, dev_iter)
        # valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, accuracy: {:.4f}"    \
                .format(epoch_time, epoch_accuracy))
        
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
                        "train_losses": train_losses},
                        os.path.join(args.log_root, "best.pth.tar"))
            # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses},
                    os.path.join(args.log_root, "conll_{}.pth.tar".format(epoch)))
        if patience_counter > args.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # test the best model
    load_file = torch.load(os.path.join(args.log_root, "best.pth.tar"),map_location=DEVICE)
    state_dict = load_file['model']
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    test_time, test_acc = test(model, test_iter)
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'best_score': best_score,
                'test_acc': test_acc,
                "epochs_count": epochs_count,
                "train_losses": train_losses},
                os.path.join(args.log_root, "best.pth.tar"))

    print("-> Test. time: {:.4f}s, accuracy: {:.4f}"    \
                .format(test_time, test_acc))


if __name__ == "__main__":
    main()