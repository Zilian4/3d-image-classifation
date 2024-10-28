import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from data_loader import get_data_list, split_data_list
from model import get_model
from seed import seed_everything
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity,RandAxisFlip

def train_fn(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        total_correct += correct
        batch_count += 1
        sample_count += len(X)
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")   
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count}

def test_fn(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
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
            pred_all.extend(torch.nn.functional.softmax(pred, dim=-1)[:, 1].cpu().numpy())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            total_correct += correct
            batch_count += 1
            sample_count += len(X)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")
    auc_score = roc_auc_score(y_all, pred_all)
    pred_all_binary = np.array(pred_all)>0.5
    precision = precision_score(y_all, pred_all_binary, zero_division=0)
    recall = recall_score(y_all, pred_all_binary, zero_division=0)
    f1 = f1_score(y_all, pred_all_binary, zero_division=0)
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count, 'auc': auc_score, 'precision': precision, 'recall': recall, 'f1': f1}, {'true': y_all, 'pred': pred_all}
    
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
    parser.add_argument("--split-seed", default=0, type=int, help="split seed")
    parser.add_argument("--resume", default="model_loss.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-s", "--seed", default=None, type=int, metavar="N", help="Seed")
    args = parser.parse_args()
    
    if args.seed:
        seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device(args.device)
    n_center = 7
    image_lists = []
    label_lists = []
    train_ds = []
    test_ds = []
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandAxisFlip(prob=0.5), RandRotate90()])
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    for c in range(n_center):
        image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
        train_image, train_label, test_image, test_label = split_data_list(image_list, label_list, ratio=args.split_ratio, seed=args.split_seed)
        print(f"Center {c+1} has {len(train_image)} training images and {len(test_image)} testing images")
        train_ds.append(ImageDataset(image_files=train_image, labels=train_label, transform=train_transforms))
        test_ds.append(ImageDataset(image_files=test_image, labels=test_label, transform=test_transforms))
    train_dataloader = DataLoader(torch.utils.data.ConcatDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = []
    for c in range(n_center):
        test_dataloader.append(DataLoader(test_ds[c], batch_size=args.batch_size, shuffle=False, num_workers=args.workers))
    n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
    n_test_ds = sum([len(test_ds[i]) for i in range(n_center)])
    
    model = get_model(out_channels = 2)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    log = {'train_loss':[], 'train_acc':[], 'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)], 'test_precision':[[] for i in range(n_center+1)], 'test_recall':[[] for i in range(n_center+1)], 'test_f1':[[] for i in range(n_center+1)]}    
    best_ind = {'loss': 0, 'acc': 0, 'auc': 0}
    for epoch in range(args.epochs):
        epoch_log = train_fn(train_dataloader, model, loss_fn, optimizer, device)
        for metric in ['loss', 'acc']:
            log['train_'+metric].append(epoch_log[metric])
        scheduler.step()
        y_all = []
        pred_all = []
        for c in range(n_center):
            epoch_log, epoch_y = test_fn(test_dataloader[c], model, loss_fn, device)
            for metric in ['loss', 'acc', 'auc', 'precision', 'recall', 'f1']:
                log['test_'+metric][c].append(epoch_log[metric])
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
        
        log['test_loss'][-1].append(sum([log['test_loss'][i][-1]*len(test_dataloader[i]) for i in range(n_center)])/n_test_dataloader)
        log['test_acc'][-1].append(sum([log['test_acc'][i][-1]*len(test_ds[i]) for i in range(n_center)])/n_test_ds)
        log['test_auc'][-1].append(roc_auc_score(y_all, pred_all))
        pred_all_binary = np.array(pred_all)>0.5
        log['test_precision'][-1].append(precision_score(y_all, pred_all_binary, zero_division=0))
        log['test_recall'][-1].append(recall_score(y_all, pred_all_binary, zero_division=0))
        log['test_f1'][-1].append(f1_score(y_all, pred_all_binary, zero_division=0))
        print(f"Epoch {epoch+1} train loss {log['train_loss'][-1]:.4f} acc {log['train_acc'][-1]:.4f}")
        for c in range(n_center):
            print(f"Center {c+1} test loss {log['test_loss'][c][-1]:.4f} acc {log['test_acc'][c][-1]:.4f} auc {log['test_auc'][c][-1]:.4f} precision {log['test_precision'][c][-1]:.4f} recall {log['test_recall'][c][-1]:.4f} f1 {log['test_f1'][c][-1]:.4f}")
        print(f"Average test loss {log['test_loss'][-1][-1]:.4f} acc {log['test_acc'][-1][-1]:.4f} auc {log['test_auc'][-1][-1]:.4f} precision {log['test_precision'][-1][-1]:.4f} recall {log['test_recall'][-1][-1]:.4f} f1 {log['test_f1'][-1][-1]:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pth"))
        with open(os.path.join(args.output_dir, "log.json"), 'w') as f:
            json.dump(log, f)
        if log['test_loss'][-1][-1] <= min(log['test_loss'][-1]):    
            best_ind['loss'] = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_loss.pth"))
        for metric, metric2 in (['acc', 'auc'], ['auc', 'acc']): # save model when metric improves or both metric and metric2 are maximized. The second condition avoids one best again but another worse
            if epoch == 0 or log['test_'+metric][-1][-1] > max(log['test_'+metric][-1][:-1]) or log['test_'+metric][-1][-1] == max(log['test_'+metric][-1][:-1]) and log['test_'+metric2][-1][-1] >= max(log['test_'+metric2][-1][:-1]):
                best_ind[metric] = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_'+metric+'.pth'))
    
    for metric in ['acc', 'auc']:
        print(f"{metric} best model reached at epoch {best_ind[metric]+1}")
        for c in range(n_center):
            print(f"Center {c+1} test loss {log['test_loss'][c][best_ind[metric]]:.4f} acc {log['test_acc'][c][best_ind[metric]]:.4f} auc {log['test_auc'][c][best_ind[metric]]:.4f} precision {log['test_precision'][-1][best_ind[metric]]:.4f} recall {log['test_recall'][-1][best_ind[metric]]:.4f} f1 {log['test_f1'][c][best_ind[metric]]:.4f}")
        print(f"Average test loss {log['test_loss'][-1][best_ind[metric]]:.4f} acc {log['test_acc'][-1][best_ind[metric]]:.4f} auc {log['test_auc'][-1][best_ind[metric]]:.4f} precision {log['test_precision'][-1][best_ind[metric]]:.4f} recall {log['test_recall'][-1][best_ind[metric]]:.4f} f1 {log['test_f1'][-1][best_ind[metric]]:.4f}")