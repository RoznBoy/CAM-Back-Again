# CAM-Back-Again 구조를 그대로 쓰는 간단한 CUB fine-tuning 코드

# train_wsol.py
# coding: utf-8
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from convnext_func import Net2Head
from replknet_func import build_model as build_replk


def build_model(model_family, num_classes=200, input_size=384):
    model_config = {
        "class_n": num_classes,
        "unit_n": 1024,
        "input_size": input_size,
        "size": 12,
        "lr": 1e-5,
        "weight_decay": 5e-4,
    }

    if model_family == "convnext":
        model_config["model_name"] = "convnext_base_384_in22ft1k"
        base = timm.create_model(model_config["model_name"], pretrained=True)
        base = nn.Sequential(*list(base.children())[:-2])
        model = nn.Sequential(
            base,
            Net2Head(model_config["class_n"],
                     model_config["unit_n"],
                     model_config["size"])
        )
    elif model_family == "replknet":
        model_config["model_name"] = "RepLKNet-31B"
        model_config["channels"] = [128, 256, 512, 1024]
        model = build_replk(model_config)  # replknet_func.py에 이미 있음
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    return model, model_config


def build_dataloaders(data_root, input_size, batch_size, aug_mode):
    normalize = transforms.Normalize((0.5,), (0.5,))

    if aug_mode == "light":
        train_tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_mode == "strong":
        train_tf = transforms.Compose([
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown aug_mode: {aug_mode}")

    test_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser("Fine-tune ConvNeXt / RepLKNet for WSOL backbone")
    p.add_argument("--model_family", type=str, required=True,
                   choices=["convnext", "replknet"])
    p.add_argument("--data_root", type=str,
                   default="datasets/cub-200-2011/CUB_200_2011/images_splitted",
                   help="CUB train/test root")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_size", type=int, default=384)
    p.add_argument("--aug_mode", type=str, default="light",
                   choices=["light", "strong"])
    p.add_argument("--save_dir", type=str, default="weights")
    p.add_argument("--exp_name", type=str, default=None)
    args = p.parse_args()

    if args.exp_name is None:
        args.exp_name = f"{args.model_family}_r{args.input_size}_e{args.epochs}_lr{args.lr}_aug{args.aug_mode}"

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = build_model(args.model_family, input_size=args.input_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    train_loader, test_loader = build_dataloaders(
        args.data_root, args.input_size, args.batch_size, args.aug_mode
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_acc = eval_top1(model, test_loader, device)
        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
              f"test_acc={test_acc:.3f}, time={dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir,
                                     f"{args.exp_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >> best updated, saved to {save_path}")

    print("Training done. Best test top-1:", best_acc)


if __name__ == "__main__":
    main()
