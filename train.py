import argparse
import numbers
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import config
from dataset import calc_data_weights, get_dataloaders
from model import resnet50
from utils import AverageMeter, TrainClock, save_args

torch.backends.cudnn.benchmark = True
LOSS_WEIGHTS = calc_data_weights()


class Session:

    def __init__(self, config, net=None):
        self.log_dir = os.path.join(config.log_dir, "bone-100")
        self.model_dir = os.path.join(config.model_dir, "bone-100")
        self.net = net
        self.best_val_acc = 0.0
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.clock = TrainClock()

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        tmp = {
            "net": self.net.state_dict(),
            "best_val_acc": self.best_val_acc,
            "clock": self.clock.make_checkpoint(),
        }
        print("\n")
        print(f"Save ckpt to {ckp_path}...")
        os.makedirs(Path(ckp_path).parent, exist_ok=True)
        torch.save(tmp, ckp_path)
        torch.save(tmp, ckp_path)
        print("Done!")
        print("\n")

    def load_checkpoint(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.net.load_state_dict(checkpoint["net"])
        self.clock.restore_checkpoint(checkpoint["clock"])
        self.best_val_acc = checkpoint["best_val_acc"]


def train_model(train_loader, model, optimizer, epoch):
    losses = AverageMeter("epoch_loss")
    accs = AverageMeter("epoch_acc")

    # ensure model is in train mode
    model.train()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        inputs = data["image"]
        labels = data["label"]
        study_type = data["meta_data"]["study_type"]
        file_paths = data["meta_data"]["file_path"]
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [
            LOSS_WEIGHTS[labels[i]][study_type[i]] for i in range(inputs.size(0))
        ]
        weights = torch.Tensor(weights).view_as(labels).to(config.device)

        outputs = model(inputs)
        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)

        # update loss metric

        # Change [64] to [64,1]
        labels = labels.unsqueeze(1)
        weights = weights.unsqueeze(1)

        # loss = F.binary_cross_entropy(
        #     outputs, labels.to(config.device).float(), weights
        # )

        criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
        loss = criterion(outputs, labels.to(config.device).float())
        losses.update(loss.item(), inputs.size(0))

        corrects = torch.sum(preds.view_as(labels) == labels.float().data)
        acc = corrects.item() / inputs.size(0)
        accs.update(acc, inputs.size(0))

        # compute gradient and do SGD step
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            loss=":{:.4f}".format(losses.avg), acc=":{:.4f}".format(accs.avg)
        )

    outspects = {"epoch_loss": losses.avg, "epoch_acc": accs.avg}
    return outspects


# original validation function
def valid_model(valid_loader, model, optimizer, epoch):
    # using model to predict, based on dataloader
    losses = AverageMeter("epoch_loss")
    accs = AverageMeter("epoch_acc")
    model.eval()

    st_corrects = {st: 0 for st in config.study_type}
    nr_stype = {st: 0 for st in config.study_type}
    study_out = {}  # study level output
    study_label = {}  # study level label
    # auc = AUCMeter()

    all_labels = []
    all_preds = []
    pos_labels = {}
    pos_preds = {}

    # evaluate the model
    pbar = tqdm(valid_loader)
    for k, data in enumerate(pbar):
        inputs = data["image"]
        labels = data["label"]
        encounter = data["meta_data"]["encounter"]
        study_type = data["meta_data"]["study_type"]
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [
            LOSS_WEIGHTS[labels[i]][study_type[i]] for i in range(inputs.size(0))
        ]
        weights = torch.Tensor(weights).view_as(labels).to(config.device)

        with torch.no_grad():

            outputs = model(inputs)
            preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)

            # Change [16] to [16,1]
            labels = labels.unsqueeze(1)
            weights = weights.unsqueeze(1)

            # update loss metric
            # loss = F.binary_cross_entropy(outputs, labels.to(config.device).float(), weights)
            criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
            loss = criterion(outputs, labels.to(config.device).float())
            losses.update(loss.item(), inputs.size(0))

            corrects = torch.sum(preds.view_as(labels) == labels.float().data)
            acc = corrects.item() / inputs.size(0)
            accs.update(acc, inputs.size(0))

            if torch.is_tensor(preds):
                preds = preds.cpu().squeeze().detach().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().squeeze().detach().numpy()
            elif isinstance(labels, numbers.Number):
                labels = np.asarray([labels])

            for i, study in enumerate(study_type):
                if study not in pos_labels:
                    pos_labels[study] = []
                if study not in pos_preds:
                    pos_preds[study] = []

                pos_labels[study].append(labels[i])
                pos_preds[study].append(preds[i])

            all_labels.extend(labels)
            all_preds.extend(preds)

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, k, len(valid_loader)))
        pbar.set_postfix(
            loss=":{:.4f}".format(losses.avg),
            acc=":{:.4f}".format(accs.avg),
            # auc=":{:.4f}".format(roc_auc_score(labels, preds)),
            # kappa=":{:.4f}".format(kappa),
        )

        for i in range(len(outputs)):
            if study_out.get(encounter[i], -1) == -1:
                study_out[encounter[i]] = [outputs[i].item()]
                study_label[encounter[i]] = labels[i].item()
                nr_stype[study_type[i]] += 1
            else:
                study_out[encounter[i]] += [outputs[i].item()]

    # study level prediction
    study_preds = {
        x: (np.mean(study_out[x]) > 0.5) == study_label[x] for x in study_out.keys()
    }

    for x in study_out.keys():
        st_corrects[x[: x.find("_")]] += study_preds[x]

    # acc for each study type
    avg_corrects = {st: st_corrects[st] / nr_stype[st] for st in config.study_type}

    total_corrects = 0
    total_samples = 0

    for st in config.study_type:
        total_corrects += st_corrects[st]
        total_samples += nr_stype[st]

    # acc for the whole dataset
    total_acc = total_corrects / total_samples

    # auc value

    # final_scores = [np.mean(study_out[x]) for x in study_out.keys()]
    # auc_output = np.array(final_scores)
    # auc_target = np.array(list(study_label.values()))
    # auc.reset()
    # auc.add(auc_output, auc_target)

    # auc_val, tpr, fpr = auc.value()
    auc_val = roc_auc_score(all_labels, all_preds)
    # Calculate Kappa coefficient
    # preds_binary = (auc_output > 0.5).astype(int)
    kappa = cohen_kappa_score(all_labels, all_preds)
    kappa_pos = {}
    for pos in pos_labels.keys():
        kappa_pos[pos] = cohen_kappa_score(pos_labels[pos], pos_preds[pos])

    pbar.set_postfix(auc=":{:.4f}".format(auc_val))
    pbar.set_postfix(kappa=":{:.4f}".format(kappa))
    print("AUC: ", auc_val)

    # Print Kappa coefficient and confidence interval
    print(f"Kappa Coefficient: {kappa}")

    for pos in pos_labels.keys():
        print(f"Kappa score {pos}: {kappa_pos[pos]}")

    torch.cuda.empty_cache()

    outspects = {
        "epoch_loss": losses.avg,
        "epoch_acc": total_acc,
        "epoch_auc": auc_val,
        "kappa": kappa,
    }

    df = pd.DataFrame(
        {
            "epoch": [epoch],
            "ACC_ELBOW": [avg_corrects["ELBOW"]],
            "ACC_FINGER": [avg_corrects["FINGER"]],
            "ACC_FOREARM": [avg_corrects["FOREARM"]],
            "ACC_HAND": [avg_corrects["HAND"]],
            "ACC_HUMERUS": [avg_corrects["HUMERUS"]],
            "ACC_SHOULDER": [avg_corrects["SHOULDER"]],
            "ACC_WRIST": [avg_corrects["WRIST"]],
            "ACC_LEG": [avg_corrects["LEG"]],
            "KAPPA_ELBOW": [kappa_pos["ELBOW"]],
            "KAPPA_FINGER": [kappa_pos["FINGER"]],
            "KAPPA_FOREARM": [kappa_pos["FOREARM"]],
            "KAPPA_HAND": [kappa_pos["HAND"]],
            "KAPPA_HUMERUS": [kappa_pos["HUMERUS"]],
            "KAPPA_SHOULDER": [kappa_pos["SHOULDER"]],
            "KAPPA_WRIST": [kappa_pos["WRIST"]],
            "KAPPA_LEG": [kappa_pos["LEG"]],
            "epoch_acc": [outspects["epoch_acc"]],
            "epoch_auc": [outspects["epoch_auc"]],
            "kappa": [kappa],
            "epoch_loss": [outspects["epoch_loss"]],
        }
    )
    csv_path = (
        "/".join(config.acc_path.split("/")[:-1])
        + "/bone_"
        + config.acc_path.split("/")[-1]
    )
    if os.path.isfile(csv_path):
        csv = pd.read_csv(csv_path)
        pd.concat([csv, df]).to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return outspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="epoch number")

    parser.add_argument(
        "-b", "--batch_size", default=8, type=int, help="mini-batch size"
    )

    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        help="initial learning rate",
    )

    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_path",
        type=str,
        required=False,
    )
    parser.add_argument("--exp_name", default=config.exp_name, type=str, required=False)

    args = parser.parse_args()
    print(args)

    config.exp_name = args.exp_name
    config.make_dir()
    save_args(args, config.log_dir)
    net = resnet50(pretrained=True)

    net = net.cuda()
    sess = Session(config, net=net)

    # get dataloader
    train_loader = get_dataloaders("train", batch_size=args.batch_size, shuffle=True)

    valid_loader = get_dataloaders("valid", batch_size=args.batch_size, shuffle=False)

    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)

    # start session
    clock = sess.clock
    tb_writer = sess.tb_writer
    # sess.save_checkpoint("start.pth.tar")

    optimizer = optim.Adam(sess.net.parameters(), args.lr)

    scheduler = ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=10, verbose=True
    )

    # start training
    for e in range(clock.epoch, args.epochs):
        train_out = train_model(train_loader, sess.net, optimizer, clock.epoch)
        valid_out = valid_model(valid_loader, sess.net, optimizer, clock.epoch)

        tb_writer.add_scalars(
            "loss",
            {"train": train_out["epoch_loss"], "valid": valid_out["epoch_loss"]},
            clock.epoch,
        )

        tb_writer.add_scalars(
            "acc",
            {"train": train_out["epoch_acc"], "valid": valid_out["epoch_acc"]},
            clock.epoch,
        )

        tb_writer.add_scalar("auc", valid_out["epoch_auc"], clock.epoch)

        tb_writer.add_scalar(
            "learning_rate", optimizer.param_groups[-1]["lr"], clock.epoch
        )
        scheduler.step(valid_out["epoch_auc"])

        if valid_out["epoch_auc"] > sess.best_val_acc:
            sess.best_val_acc = valid_out["epoch_auc"]
            sess.save_checkpoint("best_model.pth.tar")

        # Save after each 5 epoch.
        # sess.save_checkpoint("epoch{}.pth.tar".format(clock.epoch))

        sess.save_checkpoint("latest.pth.tar")

        clock.tock()


if __name__ == "__main__":
    main()
