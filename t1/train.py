import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from data_loader import get_data_list
from model import get_model
from seed import seed_everything
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity,RandAxisFlip

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}")
    return total_loss/batch_count

def test(dataloader, model, loss_fn, device):
    model.eval()
    total_loss, total_correct = 0, 0
    batch_count = 0
    sample_count = 0
    y_all = []
    pred_all = []
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Testing")
        for X, y in progress_bar:
            y_all.extend(y)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_all.extend(torch.nn.functional.softmax(pred, dim=-1).cpu().numpy())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            total_correct += correct
            batch_count += 1
            sample_count += len(X)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")    
    y_all = label_binarize(y_all, classes=[0, 1, 2])
    auc_score = roc_auc_score(y_all, pred_all, average='macro', multi_class='ovr')
    
    return total_loss/batch_count, total_correct/sample_count, auc_score
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Training.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--split-ratio", default=0.8, type=float, help="training ratio")
    parser.add_argument("--split-seed", default=0, type=float, help="split seed")
    parser.add_argument("--resume", default="model_loss.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-s", "--seed", default=None, type=int, metavar="N", help="Seed")
    args = parser.parse_args()
    
    if args.seed:
        seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device(args.device)
    image_list, label_list = get_data_list(root=args.data_path, t = args.t)
    split = int(np.floor(len(image_list) * args.split_ratio))
    indices = np.random.default_rng(seed=args.split_seed).permutation(len(image_list))
    train_idx, test_idx = list(indices[:split]), list(indices[split:])
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandAxisFlip(prob=0.5), RandRotate90()])
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    train_ds = ImageDataset(image_files=[image_list[i] for i in train_idx], labels=[label_list[i] for i in train_idx], transform=train_transforms)
    test_ds = ImageDataset(image_files=[image_list[i] for i in test_idx], labels=[label_list[i] for i in test_idx], transform=test_transforms)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = get_model()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    log = {'train_loss':[], 'test_loss':[], 'test_acc':[], 'test_auc':[]}
    for epoch in range(args.epochs):
        log['train_loss'].append(train(train_dataloader, model, loss_fn, optimizer, device))
        scheduler.step()
        loss, acc, auc = test(test_dataloader, model, loss_fn, device)
        log['test_loss'].append(loss)
        log['test_acc'].append(acc)
        log['test_auc'].append(auc)
        print(f"Epoch {epoch} train loss {log['train_loss'][-1]:.4f} test loss {log['test_loss'][-1]:.4f} test acc {log['test_acc'][-1]:.4f}  test auc {log['test_auc'][-1]:.4f}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pth"))
        with open(os.path.join(args.output_dir, "log.json"), 'w') as f:
            json.dump(log, f)
        if log['test_loss'][-1] <= min(log['test_loss']):    
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_loss.pth"))
        if log['test_acc'][-1] >= max(log['test_acc']):    
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_acc.pth"))
        if log['test_auc'][-1] >= max(log['test_auc']):    
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_auc.pth"))
    print(f"Acc best model test acc {max(log['test_acc']):.4f} test auc {log['test_auc'][np.argmax(log['test_acc'])]:.4f}")
    print(f"Auc best model test acc {log['test_acc'][np.argmax(log['test_auc'])]:.4f} test auc {max(log['test_auc']):.4f}")