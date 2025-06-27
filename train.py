import os
import argparse
import random
import shutil
import sys
import yaml
from easydict import EasyDict
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer

from net.cps import CPS


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, config):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": config.train_params.learning_rate},
        "aux": {"type": "Adam", "lr": config.train_params.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, tb_writer
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        
            tb_writer.add_scalar('Train_Loss', out_criterion["loss"].item(), epoch * len(train_dataloader) + i)
            tb_writer.add_scalar('Train_MSE_Loss', out_criterion["mse_loss"].item(), epoch * len(train_dataloader) + i)
            tb_writer.add_scalar('Train_bpp_Loss', out_criterion["bpp_loss"].item(), epoch * len(train_dataloader) + i)
            tb_writer.add_scalar('Train_Aux_Loss', aux_loss.item(), epoch * len(train_dataloader) + i)


def test_epoch(epoch, test_dataloader, model, criterion, tb_writer):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    tb_writer.add_scalar('Test_Loss', loss.avg, epoch)
    tb_writer.add_scalar('Test_MSE_Loss', mse_loss.avg, epoch)
    tb_writer.add_scalar('Test_bpp_Loss', bpp_loss.avg, epoch)
    tb_writer.add_scalar('Test_Aux_Loss', aux_loss.avg, epoch)

    return loss.avg


def save_checkpoint(state, is_best, root_path):
    save_path = os.path.join(root_path, f"checkpoint.pth.tar")
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(root_path, "checkpoint_best_loss.pth.tar")
        shutil.copyfile(save_path, best_path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    
    parser.add_argument("--config", type=str, help="Experiment config")
    parser.add_argument("--start-fresh", action="store_true", help="Reset training state.")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(config.train_params.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(config.train_params.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(config.train_data.data_path, split="train", transform=train_transforms)
    test_dataset = ImageFolder(config.train_data.data_path, split="test", transform=test_transforms)

    device = "cuda" if config.train_data.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_data.batch_size,
        num_workers=config.train_data.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train_data.test_batch_size,
        num_workers=config.train_data.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = CPS(config.train_params.N)
    net = net.to(device)

    if config.train_data.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, config)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=config.train_params.lmbda)

    experiment_path = config.train_data.experiment_path
    os.makedirs(experiment_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=experiment_path)

    last_epoch = 0
    if len(config.train_data.use_checkpoint_path) != 0:  # load from previous checkpoint
        print("Loading", config.train_data.use_checkpoint_path)
        checkpoint = torch.load(config.train_data.use_checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if not args.start_fresh:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, config.train_data.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            config.train_params.clip_max_norm,
            tb_writer
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, tb_writer)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
            config.train_data.experiment_path
        )


if __name__ == "__main__":
    main(sys.argv[1:])