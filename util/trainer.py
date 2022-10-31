import logging
import os
import os.path as path
import typing
from collections import OrderedDict
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import torch
import torchvision
from dataset import CIFAR10
from model import (LogisticsModel, resnet50)
from sklearn.metrics import accuracy_score

try:
    if os.name == "POSIX":
        import matplotlib
        matplotlib.use("Agg")
finally:
    from matplotlib import pyplot as plt


TORCH_SUMMARY_STRING = True
try:
    from torchsummary import summary_string
except ImportError:
    TORCH_SUMMARY_STRING = False


__all__ = ["Trainer","PDMMTrainer"]


class Trainer(object):
    def __init__(self, outdir: str, seed: int,
                 datadir: str = "./data",
                 dataset_name: str = "cifar10",
                 model_name: str = "resnet50",
                 group_channels: int = 32,
                 drop_rate: float = 0.1, last_drop_rate: float = 0.5,
                 data_init_seed: int = 11,
                 model_init_seed: int = 13,
                 train_data_length: int = 12800,
                 cuda: bool = True,
                 cuda_device_no: int = 0,
                 **kwargs):
        self.__outdir__ = outdir
        self.__seed__ = seed
        self.__data_init_seed__ = data_init_seed
        self.__datadir__ = datadir
        self.__dataset_name__ = dataset_name
        self.__datasets__ = None
        self.__optim__ = None
        self.__train_data_length__ = train_data_length
        self.__sleep_factor__ = 0.0

        if cuda:
            if not torch.cuda.is_available():
                raise ValueError("CUDA device is not available!!!")
            self.__device__ = "cuda:%d" % (cuda_device_no)
        else:
            self.__device__ = "cpu"
        self.__cuda__ = cuda

        os.makedirs(self.outdir, exist_ok=True)

        self.__model_name__ = model_name
        self.__model__ = self.build_model(model_init_seed=model_init_seed,
                                          group_channels=group_channels,
                                          drop_rate=drop_rate,
                                          last_drop_rate=last_drop_rate)

        self.model.to(self.device)
        #logging.info("device: %s %d/%d, %s" % (self.device,
        #                                       torch.cuda.current_device(),
        #                                       torch.cuda.device_count(),
        #                                       torch.cuda.get_device_name(torch.cuda.current_device())))

    @property
    def outdir(self): return self.__outdir__
    @property
    def seed(self): return self.__seed__
    @property
    def data_init_seed(self): return self.__data_init_seed__
    @property
    def datadir(self): return self.__datadir__
    @property
    def cuda(self): return self.__cuda__
    @property
    def device(self): return self.__device__
    @property
    def dataset_name(self): return self.__dataset_name__

    @property
    def datasets(self):
        if self.__datasets__ is None:
            self.__datasets__ = self.load_datas(
                train_data_length=self.train_data_length)
        return self.__datasets__

    @property
    def train_data_length(self): return self.__train_data_length__

    @property
    def train_log(self): return path.join(self.outdir, "train.log")
    @property
    def eval_log(self): return path.join(self.outdir, "eval.log")
    @property
    def state_dict_file(self): return path.join(self.outdir, "model.pth")

    @property
    def score_filenames(self): return {k: "scores_%s.csv" % k
                                       for k in self.datasets}

    @property
    def fig_prefix(self): return ""
    @property
    def fig_suffix(self): return ""

    @property
    def sleep_factor(self): return self.__sleep_factor__
    @sleep_factor.setter
    def sleep_factor(self, val): self.__sleep_factor__ = float(val)

    @property
    def model(self) -> torch.nn.Module: return self.__model__
    @property
    def model_name(self) -> str: return self.__model_name__

    def build_model(self, model_init_seed: int = 32,
                    group_channels: int = 64,
                    drop_rate: float = 0.1,
                    last_drop_rate: float = 0.5,
                    **kwargs) -> torch.nn.Module:
        torch.manual_seed(model_init_seed)

        if self.dataset_name == "cifar10":
            image_shape = (3, 32, 32)
            num_classes = 10
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        if self.model_name == "resnet50":
            from model import resnet50
            model = resnet50(in_channel=image_shape[0],
                             drop_rate=drop_rate,
                             last_drop_rate=last_drop_rate,
                             num_classes=num_classes,
                             norm_layer=torch.nn.GroupNorm,
                             group_channels=group_channels)
        elif self.model_name == "logistics":
            model = LogisticsModel(in_features=np.prod(image_shape),
                                   num_classes=num_classes)
        else:
            raise ValueError("Unknonw model name: %s" % self.model_name)

        if TORCH_SUMMARY_STRING:
            # print to log
            summary, _ = summary_string(model, image_shape,
                                        device="cpu",
                                        dtypes=[torch.float]*len(image_shape))
            logging.info("===== Model Summary =====\n%s" % summary)

        for n, l in model.named_modules():
            if isinstance(l, torch.nn.GroupNorm):
                logging.info("%s: %s, groups: %d, channels: %d, eps: %f, affine: %s" % (
                    n, str(type(l)), l.num_groups, l.num_channels, l.eps, l.affine))
            else:
                logging.info("%s: %s" % (n, str(type(l))))

        model_size = 0
        b = 32/8
        for n, p in model.named_parameters():
            param_size = np.prod(p.size())*b
            logging.info("parameter size: %s %dB" % (n, param_size))
            model_size += param_size
        logging.info("Model: %s size: %.3fMiB" %
                     (self.model_name, model_size/(1024*1024)))

        return model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optim__

    def load_datas(self, indices: typing.List[int] = None, train_data_length: int = None):
        transforms = [torchvision.transforms.ToTensor()]
        if self.dataset_name == "cifar10":
            transforms.append(torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transforms = torchvision.transforms.Compose(transforms)

        if self.dataset_name == "cifar10":
            trains = CIFAR10(self.datadir, train=True,
                             download=True, transform=transforms,
                             indices=indices, data_length=train_data_length,
                             shuffle=True)
            vals = CIFAR10(self.datadir, train=True,
                           download=True, transform=transforms)
            tests = CIFAR10(self.datadir, train=False,
                            download=True, transform=transforms)
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        return OrderedDict([("train", trains), ("val", vals), ("test", tests)])

    def build_criterion(self, reduction: str = "mean", *args, **kwargs):
        return torch.nn.CrossEntropyLoss(reduction=reduction)

    def build_optimizer(self, optimizer: str = "sgd", args: typing.Dict = {}):
        if optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), **args)
        elif optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), **args)
        else:
            raise ValueError("Unknonw optimizer: %s" % (optimizer))

    def epoch(self, dataloader, criterion, optimizer,
              l2_lambda: float = 0.01, **kwargs):
        losses = OrderedDict([("loss", 0.0)])
        ndata, nbatch = 0, 0
        l2_loss = 0.0
        total_exchange_time = 0.0
        for i, (x, y) in enumerate(dataloader):
            x = x.float().to(self.device)
            y = y.to(self.device)
            o = self.model(x)

            loss = criterion(o, y).sum(dim=0)
            l2 = torch.tensor(0.0).to(self.device)
            for param in self.model.parameters():
                l2 += ((0.5*torch.sum(param**2)))
            if loss.requires_grad and optimizer is not None:
                optimizer.zero_grad()
                update_loss = loss.sum() + (l2 * l2_lambda)
                update_loss.backward()
                start = datetime.now()
                optimizer.step()
                total_exchange_time += (datetime.now() - start).total_seconds()
            loss = loss.sum().detach().cpu().item()
            losses["loss"] += loss
            l2_loss = l2.detach().cpu().item()

            ndata += len(x)
            nbatch += 1
            logging.debug("[%04d] loss: %.3f, l2 loss: %.3f" %
                          (i, loss, l2_loss))
            del x, y, o, loss
        if criterion.reduction == "mean":
            losses = OrderedDict([(k, v/nbatch) for k, v in losses.items()])
        else:  # reduction is sum or none
            losses = OrderedDict([(k, v/ndata) for k, v in losses.items()])
        losses["l2"] = l2_loss
        losses["excange_proc"] = total_exchange_time
        return losses

    def metric(self, dataloader, batch_size: int = 64, output: str = None,
               **kwargs):
        scores, labels = [], []
        for x, y in dataloader:
            x = x.float().to(self.device)
            o = self.model(x)
            labels.append(y.detach().cpu().numpy())
            scores.append(o.detach().cpu().numpy())
            del x, y, o
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)

        probs = 1/(1+np.exp(-scores))  # convert score to probavility.
        preds = np.argmax(probs, axis=1)
        metrics = OrderedDict([("acc", accuracy_score(labels, preds))])

        if output is not None:
            # columns: true label, predict label, sequence of prob per each label...
            outputs = OrderedDict([("label", labels)] +
                                  [("predict", preds)] +
                                  [(f"prob_{i}", probs[:, i]) for i in range(probs.shape[1])])
            outputs = pd.DataFrame(outputs)
            outputs.to_csv(output, index=False)
        return metrics

    def dispose(self):
        import gc
        if hasattr(self.__optim__, "notice_train_ending"):
            self.__optim__.notice_train_ending()
        del self.__optim__
        gc.collect()
        logging.info("GC: check garbage %s" % (str(gc.garbage)))
        gc.collect()

    def train(self, epochs: int, batch_size: int,
              optimizer: str = "sgd",
              optim_args: typing.Dict = {},
              scheduler_name: str = None,
              scheduler_args: list = [],
              scheduler_kwargs: dict = {},
              adameps: float = 1e-8,
              **kwargs):
        loaders = OrderedDict([(k, torch.utils.data.DataLoader(v,
                                                               batch_size=batch_size,
                                                               shuffle=(k == "train")))
                               for k, v in self.datasets.items()])

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        criterion = self.build_criterion(**kwargs)
        self.__optim__ = self.build_optimizer(optimizer=optimizer,
                                              args=optim_args)
        optimizer = self.__optim__
        if scheduler_name is not None and hasattr(torch.optim.lr_scheduler, scheduler_name):
            last_epoch = scheduler_kwargs["last_epoch"] if "last_epoch" in scheduler_kwargs else -1
            if not last_epoch == -1 and not scheduler_name == "ReduceLROnPlateau":
                for group in optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
            scheduler_ctor = getattr(torch.optim.lr_scheduler, scheduler_name)
            scheduler = scheduler_ctor(optimizer,
                                       *scheduler_args, **scheduler_kwargs)
            logging.info("Setup scheduler: %s" % (str(scheduler)))
        else:
            scheduler = None

        if hasattr(optimizer, "edges"):
            num_edges = len(
                [edge for edge in optimizer.edges() if edge.is_connected])
        else:
            num_edges = 0
        logging.debug("[%4d/%4d] connecting edge count: %d" % (
            0, epochs, num_edges))

        random_state = np.random.RandomState(self.seed)
        with open(self.train_log, "wt") as f:
            for epoch in range(1, epochs+1):
                logs = OrderedDict([("epoch", epoch)])
                start = datetime.now()

                epoch_losses, epoch_metrics = OrderedDict(), OrderedDict()
                # train
                self.model.train()
                epoch_losses["train"] = self.epoch(loaders["train"],
                                                   criterion,
                                                   optimizer,
                                                   **kwargs)
                train_proc = (datetime.now() - start).total_seconds()
                if scheduler is not None and hasattr(scheduler, "step"):
                    scheduler.step()

                if hasattr(optimizer, "edges"):
                    curr_num_edges = len(
                        [edge for edge in optimizer.edges() if edge.is_connected])
                    logging.info("[%4d/%4d] connecting edge count: %d" % (
                        epoch, epochs, curr_num_edges))
                else:
                    curr_num_edges = 0

                # unset data augmentator
                if isinstance(loaders["train"].dataset, torch.utils.data.ConcatDataset):
                    for d in loaders["train"].dataset.datasets:
                        d.data_aug = None
                else:
                    loaders["train"].dataset.data_aug = None

                # validation and test
                start = datetime.now()
                self.model.eval()
                with torch.no_grad():
                    for k, loader in loaders.items():
                        if not k == "train":
                            epoch_losses[k] = self.epoch(loader, criterion, None,
                                                         **kwargs)
                        epoch_metrics[k] = self.metric(loader,
                                                       batch_size=batch_size,
                                                       output=None,
                                                       **kwargs)
                eval_proc = (datetime.now() - start).total_seconds()

                # merge train, test loss and metrics
                for phase, losses in epoch_losses.items():
                    for k, v in losses.items():
                        if "loss" in k or "proc" in k:
                            logs["%s_%s" % (phase, k)] = v
                        else:
                            logs["%s_%s_loss" % (phase, k)] = v
                for phase, metrics in epoch_metrics.items():
                    for k, v in metrics.items():
                        logs["%s_%s" % (phase, k)] = v
                if hasattr(optimizer, "diff"):
                    diff = optimizer.diff()
                    if isinstance(diff, torch.Tensor):
                        diff = diff.detach().cpu().item()
                else:
                    diff = 0.0
                logs["diff"] = diff
                logs["train_proc"] = train_proc
                logs["eval_proc"] = eval_proc
                if hasattr(optimizer, "get_communication_time"):
                    logs["comm_proc"] = optimizer.get_communication_time()
                else:
                    logs["comm_proc"] = 0.0
                logs["timestamp"] = datetime.now()

                # write log file.
                logging.info("[%4d/%4d] loss: (train=%s, val=%s, test=%s, l2=%s), acc: (train=%s, val=%s, test=%s), diff=%.8f, proc(train=%.3fsec, eval=%.3fsec)" % (
                    epoch, epochs,
                    *["%.3f" % logs[k] for k in ["train_loss", "val_loss", "test_loss", "train_l2_loss",
                                                 "train_acc", "val_acc", "test_acc"]],
                    diff, train_proc, eval_proc))

                sleep_proc = (train_proc * (0.5 + 0.5 *
                                            self.sleep_factor * random_state.rand()))
                logs["sleep_proc"] = sleep_proc if self.sleep_factor > 0.0 else 0.0
                if self.sleep_factor > 0.0 and sleep_proc > 0.0:
                    logging.info("[%4d/%4d] Sleep %fsec" %
                                 (epoch, epochs, sleep_proc))
                    sleep(sleep_proc)

                if num_edges > curr_num_edges:
                    logging.info("[%4d/%4d] found edge disconnection: %d -> %d. finished train." % (
                        epoch, epochs, num_edges, curr_num_edges))
                    break
                num_edges = np.max([num_edges, curr_num_edges])

                # write train log.
                if epoch == 1:
                    f.write(",".join(logs.keys()))
                    f.write("\n")
                f.write(",".join(map(str, logs.values())))
                f.write("\n")

    def evaluate(self, batch_size: int,
                 optimizer: str = "sgd", lr: float = 0.01,
                 **kwargs):
        dataloaders = OrderedDict([(k, torch.utils.data.DataLoader(v,
                                                                   batch_size=batch_size,
                                                                   shuffle=False))
                                   for k, v in self.datasets.items()])

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        criterion = self.build_criterion(**kwargs)
        self.model.eval()

        outdir = self.outdir
        os.makedirs(outdir, exist_ok=True)
        with open(self.eval_log, "wt") as f, torch.no_grad():
            logs = OrderedDict([("epoch", 0)])
            start = datetime.now()
            eval_losses, eval_metrics = OrderedDict(), OrderedDict()
            for k, loader in dataloaders.items():
                eval_losses[k] = self.epoch(loader, criterion, None,
                                            **kwargs)

                eval_metrics[k] = self.metric(loader,
                                              batch_size=batch_size,
                                              output=path.join(outdir,
                                                               self.score_filenames[k]),
                                              **kwargs)
            proc = (datetime.now() - start).total_seconds()

            # merge train, test loss and metrics
            for phase, losses in eval_losses.items():
                for k, v in losses.items():
                    if k == "loss":
                        logs["%s_loss" % (phase)] = v
                    else:
                        logs["%s_%s_loss" % (phase, k)] = v
            for phase, metrics in eval_metrics.items():
                for k, v in metrics.items():
                    logs["%s_%s" % (phase, k)] = v
            logs["proc"] = proc

            # write header.
            f.write(",".join(logs.keys()))
            f.write("\n")
            # write eval log.
            f.write(",".join(map(str, logs.values())))
            f.write("\n")
            logging.info("[EVAL] loss: (train=%s, val=%s, test=%s, l2=%s), acc: (train=%s, val=%s, test=%s), proc=%.3fsec" % (
                *["%.3f" % logs[k] for k in ["train_loss", "val_loss", "test_loss", "train_l2_loss",
                                             "train_acc", "val_acc", "test_acc"]],
                proc))

    def save_state_dict(self, **kwrags):
        torch.save(self.model.state_dict(), self.state_dict_file)

    def plot_figs(self, **kwargs):
        outdir = path.join(self.outdir, "imgs")
        os.makedirs(outdir, exist_ok=True)

        # read train log
        train_log = pd.read_csv(self.train_log)
        # read predict files.
        pred_vals = OrderedDict([(k, pd.read_csv(
            path.join(self.outdir, self.score_filenames[k]))) for k in self.datasets.keys()])

        # plot loss, acc
        for metric in ["loss", "acc", "l2_loss"]:
            figname = path.join(outdir, "%s%s%s.png" % (
                self.fig_prefix, metric, self.fig_suffix))
            for k in self.datasets.keys():
                plt.plot(train_log["%s_%s" % (k, metric)].values, label=k)
            plt.title(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()


class PDMMTrainer(Trainer):
    def __init__(self, outdir: str,
                 nodename: str, edges: typing.Dict, hosts: typing.Dict,
                 seed: int,
                 datadir: str = "./datas",
                 dataset_name: str = "cifar10",
                 model_name: str = "resnet32",
                 group_channels: int = 32,
                 drop_rate: float = 0.1,
                 last_drop_rate: float = 0.5,
                 data_init_seed: int = 11,
                 model_init_seed: int = 13,
                 train_data_length: int = 12800,
                 cuda: bool = True,
                 cuda_device_no: int = 0,
                 **kwargs):
        super(PDMMTrainer, self).__init__(outdir, seed,
                                          datadir=datadir,
                                          dataset_name=dataset_name,
                                          model_name=model_name,
                                          group_channels=group_channels,
                                          drop_rate=drop_rate, last_drop_rate=last_drop_rate,
                                          data_init_seed=data_init_seed,
                                          model_init_seed=model_init_seed,
                                          train_data_length=train_data_length,
                                          cuda=cuda,
                                          cuda_device_no=cuda_device_no,
                                          **kwargs)
        self.__nodename__ = nodename
        self.__nodeindex__ = sorted([n["name"] for n in hosts]).index(nodename)
        self.__hosts__ = hosts
        self.__edges__ = edges
        self.__common_classes__ = 0
        self.__node_per_classes__ = 8
        self.__class_inbalanced_lambda__ = 1.0
        self.__data_inbalanced_lambda__ = 0.2
        self.__nshift_of_nodes__ = 4
        self.__data_split_mode__ = "split"

    @property
    def train_log(self): return path.join(self.outdir,
                                          "%s.train.log" % (self.nodename))

    @property
    def eval_log(self): return path.join(self.outdir,
                                         "%s.eval.log" % (self.nodename))

    @property
    def state_dict_file(self): return path.join(self.outdir,
                                                "%s.model.pth" % (self.nodename))

    @property
    def model_file(self): return path.join(self.outdir,
                                           "%s.model.pt" % (self.nodename))

    @property
    def score_filenames(self): return {k: "%s.scores_%s.csv" % (self.nodename, k)
                                       for k in self.datasets}

    @property
    def fig_prefix(self): return "%s." % (self.nodename)
    @property
    def fig_suffix(self): return ""

    @property
    def nodename(self): return self.__nodename__
    @property
    def nodeindex(self): return self.__nodeindex__
    @property
    def hosts(self): return self.__hosts__
    @property
    def edges(self): return self.__edges__
    @property
    def common_classes(self): return self.__common_classes__
    @property
    def node_per_classes(self): return self.__node_per_classes__
    @property
    def nshift_of_nodes(self): return self.__nshift_of_nodes__
    @property
    def data_split_mode(self): return self.__data_split_mode__
    @property
    def class_inbalanced_lambda(self): return self.__class_inbalanced_lambda__
    @property
    def data_inbalanced_lambda(self): return self.__data_inbalanced_lambda__

    def build_optimizer(self,
                        optimizer: str = "PdmmISVR",
                        args: typing.Dict = {}):
        round_cnt = args["round"]
        del args["round"]
        if optimizer in ["PdmmISVR", "AdmmISVR"]:
            from edgecons import Ecl
            return Ecl(self.nodename,
                       round_cnt,
                       self.edges,
                       self.hosts,
                       self.model,
                       device=self.device,
                       **args)
        elif optimizer == "DSGD":
            from optimizer.dsgd import DSGD
            return DSGD(self.nodename,
                        round_cnt,
                        self.edges,
                        self.hosts,
                        self.model,
                        device=self.device,
                        **args)
        else:
            raise ValueError("Unknonw optimizer: %s" % (optimizer))

    def load_datas(self,
                   indices: typing.List[int] = None,
                   train_data_length: int = None):
        if self.dataset_name == "cifar10":
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif self.dataset_name == "fashion":
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        if self.data_split_mode == "split":
            return self.split_load_datas(indices=indices, train_data_length=train_data_length)
        elif self.data_split_mode == "same":
            return super(PdmmTrainer, self).load_datas(indices=indices, train_data_length=train_data_length)
        elif self.data_split_mode == "split_class_data_same":
            return self.split_class_data_same_load_datas(indices=indices, train_data_length=train_data_length)
        elif self.data_split_mode == "split_class_data_even":
            return self.split_class_data_even_load_datas(indices=indices, train_data_length=train_data_length)
        else:
            raise ValueError("Unsupported data split mode: %s" %
                             (self.data_split_mode))

    def split_load_datas(self,
                         indices: typing.List[int] = None,
                         train_data_length: int = None):
        if self.dataset_name == "cifar10":
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif self.dataset_name == "fashion":
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        assert num_classes - self.common_classes - self.node_per_classes > 0
        class_indices = list(range(num_classes))
        common_class_indices = class_indices[:self.common_classes]
        split_class_indices = class_indices[self.common_classes:]

        node_to_class_idx = [[] for i in range(len(self.hosts))]
        class_to_node_idx = [[] for i in range(num_classes)]
        class_data_weights = []
        for class_idx in common_class_indices:
            for node_idx, node_classes in enumerate(node_to_class_idx):
                node_classes.append(class_idx)
                class_to_node_idx[class_idx].append(node_idx)
            class_data_weights.append(
                np.array([1.0/len(self.hosts)] * len(self.hosts)))

        if self.node_per_classes > 0:
            assert self.node_per_classes < len(split_class_indices)
            assert self.nshift_of_nodes > 0
            np.random.shuffle(split_class_indices)
            for node_idx, node_indices in enumerate(node_to_class_idx):
                sidx = (node_idx * self.nshift_of_nodes) % (len(split_class_indices))
                eidx = sidx + self.node_per_classes
                class_indices = split_class_indices[sidx:eidx]
                if eidx > len(split_class_indices):
                    class_indices += split_class_indices[:(
                        eidx-len(split_class_indices))]
                node_indices.extend(class_indices)
                for class_idx in class_indices:
                    class_to_node_idx[class_idx].append(node_idx)

        for node_idxs in class_to_node_idx[len(class_data_weights):]:
            w = (np.random.rand(len(node_idxs)) * self.data_inbalanced_lambda +
                 (1.0 - self.data_inbalanced_lambda))
            w /= np.sum(w)
            class_data_weights.append(w)

        # logging
        for node_idx, class_idxs in enumerate(node_to_class_idx):
            logging.info("node%d: [%s]" %
                         (node_idx, ",".join(map(str, class_idxs))))
        for class_idx, node_idxs in enumerate(class_to_node_idx):
            logging.info("class %d, [%s]" % (class_idx,
                                             ",".join(["node%d:%.3f" % (n, class_data_weights[class_idx][i])
                                                       for i, n in enumerate(node_idxs)])))

        class_data_idxs = [np.where(targets == i)[0]
                           for i in range(num_classes)]
        node_data_indices = [[] for _ in range(len(self.hosts))]
        for class_idx, (node_idxs, node_weights, class_datas) in enumerate(zip(class_to_node_idx,
                                                                               class_data_weights,
                                                                               class_data_idxs)):
            ndata_of_nodes = (node_weights * len(class_datas)).astype(np.int)
            offsets = []
            offset = 0
            for n in ndata_of_nodes:
                offset += n
                offsets.append(offset)
            offsets[-1] = len(class_datas)  
            offset = 0
            for node_idx, end_offset in zip(node_idxs, offsets):
                node_data_indices[node_idx].append(
                    class_datas[offset:end_offset])
                offset = end_offset

        node_data_indices = [np.concatenate(d, axis=0)
                             for d in node_data_indices]

        # write train data summary...
        for node_idx, data_indices in enumerate(node_data_indices):
            node_targets = targets[data_indices]
            classes, counts = np.unique(node_targets, return_counts=True)
            logging.info("train data node%d: %s" % (node_idx,
                                                    ",".join(["%d:%d" % (i, n) for i, n in zip(classes, counts)])))
        node_indices = node_data_indices[self.nodeindex]
        datasets = super(PDMMTrainer, self).load_datas(indices=node_indices,
                                                       train_data_length=train_data_length)
        return datasets

    def split_class_data_same_load_datas(self,
                                         indices: typing.List[int] = None,
                                         train_data_length: int = None):
        if self.dataset_name == "cifar10":
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif self.dataset_name == "fashion":
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        assert num_classes - self.common_classes - self.node_per_classes > 0
        class_indices = list(range(num_classes))
        common_class_indices = class_indices[:self.common_classes]
        split_class_indices = class_indices[self.common_classes:]

        node_to_class_idx = [[] for i in range(len(self.hosts))]
        class_to_node_idx = [[] for i in range(num_classes)]
        # class_data_weights = []
        for class_idx in common_class_indices:
            for node_idx, node_classes in enumerate(node_to_class_idx):
                node_classes.append(class_idx)
                class_to_node_idx[class_idx].append(node_idx)
            # class_data_weights.append(
            #     np.array([1.0/len(self.hosts)] * len(self.hosts)))

        if self.node_per_classes > 0:
            assert self.node_per_classes < len(split_class_indices)
            assert self.nshift_of_nodes > 0
            np.random.shuffle(split_class_indices)
            for node_idx, node_indices in enumerate(node_to_class_idx):
                sidx = (node_idx * self.nshift_of_nodes) % (len(split_class_indices))
                eidx = sidx + self.node_per_classes
                class_indices = split_class_indices[sidx:eidx]
                if eidx > len(split_class_indices):
                    class_indices += split_class_indices[:(
                        eidx-len(split_class_indices))]
                node_indices.extend(class_indices)
                for class_idx in class_indices:
                    class_to_node_idx[class_idx].append(node_idx)

        # logging
        for node_idx, class_idxs in enumerate(node_to_class_idx):
            logging.info("node%d: [%s]" %
                         (node_idx, ",".join(map(str, class_idxs))))
        for class_idx, node_idxs in enumerate(class_to_node_idx):
            logging.info("class %d, [%s]" % (class_idx,
                                             ",".join(["node%d:%.3f" % (n, 1.0)
                                                       for i, n in enumerate(node_idxs)])))

        class_data_idxs = [np.where(targets == i)[0]
                           for i in range(num_classes)]
        node_data_indices = [[] for _ in range(len(self.hosts))]
        for node_idxs, class_datas in zip(class_to_node_idx,
                                          class_data_idxs):
            for node_idx in node_idxs:
                node_data_indices[node_idx].append(class_datas)

        node_data_indices = [np.concatenate(d, axis=0)
                             for d in node_data_indices]

        # write train data summary...
        for node_idx, data_indices in enumerate(node_data_indices):
            node_targets = targets[data_indices]
            classes, counts = np.unique(node_targets, return_counts=True)
            logging.info("train data node%d: %s" % (node_idx,
                                                    ",".join(["%d:%d" % (i, n) for i, n in zip(classes, counts)])))
        node_indices = node_data_indices[self.nodeindex]
        datasets = super(PdmmTrainer, self).load_datas(indices=node_indices,
                                                       train_data_length=train_data_length)
        return datasets

    def split_class_data_even_load_datas(self,
                                         indices: typing.List[int] = None,
                                         train_data_length: int = None):
        dataset_type = self.dataset_type(self.dataset_name)
        if dataset_type == 0:
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif dataset_type == 1:
            trains = torchvision.datasets.EMNIST(self.datadir, "letters", train=True,
                                                 download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif dataset_type == 2:
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        assert num_classes - self.common_classes - self.node_per_classes > 0
        class_indices = list(range(num_classes))
        common_class_indices = class_indices[:self.common_classes]
        split_class_indices = class_indices[self.common_classes:]

        node_to_class_idx = [[] for i in range(len(self.hosts))]
        class_to_node_idx = [[] for i in range(num_classes)]
        # class_data_weights = []
        for class_idx in common_class_indices:
            for node_idx, node_classes in enumerate(node_to_class_idx):
                node_classes.append(class_idx)
                class_to_node_idx[class_idx].append(node_idx)

        if self.node_per_classes > 0:
            assert self.node_per_classes < len(split_class_indices)
            assert self.nshift_of_nodes > 0
            np.random.shuffle(split_class_indices)
            for node_idx, node_indices in enumerate(node_to_class_idx):
                sidx = (node_idx * self.nshift_of_nodes) % (len(split_class_indices))
                eidx = sidx + self.node_per_classes
                class_indices = split_class_indices[sidx:eidx]
                if eidx > len(split_class_indices):
                    class_indices += split_class_indices[:(
                        eidx-len(split_class_indices))]
                node_indices.extend(class_indices)
                for class_idx in class_indices:
                    class_to_node_idx[class_idx].append(node_idx)

        # logging
        for node_idx, class_idxs in enumerate(node_to_class_idx):
            logging.info("node%d: [%s]" %
                         (node_idx, ",".join(map(str, class_idxs))))
        for class_idx, node_idxs in enumerate(class_to_node_idx):
            logging.info("class %d, [%s]" % (class_idx,
                                             ",".join(["node%d:%.3f" % (n, 1.0)
                                                       for i, n in enumerate(node_idxs)])))

        class_data_idxs = [np.where(targets == i)[0]
                           for i in range(num_classes)]
        node_data_indices = [[] for _ in range(len(self.hosts))]
        for node_idxs, class_datas in zip(class_to_node_idx,
                                          class_data_idxs):
            splited = np.array_split(class_datas, len(node_idxs))
            for node_idx, n in zip(node_idxs, splited):
                node_data_indices[node_idx].append(n)

        node_data_indices = [np.concatenate(d, axis=0)
                             for d in node_data_indices]

        # write train data summary...
        for node_idx, data_indices in enumerate(node_data_indices):
            node_targets = targets[data_indices]
            classes, counts = np.unique(node_targets, return_counts=True)
            logging.info("train data node%d: %s" % (node_idx,
                                                    ",".join(["%d:%d" % (i, n) for i, n in zip(classes, counts)])))
        node_indices = node_data_indices[self.nodeindex]
        datasets = super(PdmmTrainer, self).load_datas(indices=node_indices,
                                                       train_data_length=train_data_length)
        return datasets
