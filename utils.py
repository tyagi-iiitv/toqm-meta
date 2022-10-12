#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import asyncio
import itertools
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, OrderedDict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from fblearner.flow.projects.users.anjultyagi.mbv2_toqm import (
    HardQuantizeConv,
    LsqQuan as lsqquan_toqm,
    quan_bn,
)
from manifold.blobstore.blobstore.types import DirectoryEntry
from manifold.clients.python import DirectoryEntryType, ManifoldClient
from on_device_ai.fast_nas.searchspace.choice import Choice
from on_device_ai.fast_nas.searchspace.uniform import Uniform
from on_device_ai.tools.data.imagenet.loader import build_imagenet_dataloader
from pytorch_lightning import Callback
from torch.ao.pruning import fqn_to_module
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from torchvision import models, transforms


@dataclass(frozen=True)
class Params:
    r"""
    These values are set to default based on the training parameters of
    pytorch models available at
    https://github.com/pytorch/vision/tree/main/references/classification
    """
    # Batch size. Generally 32 is good per GPU. Set to 32*num_gpus in your experiment
    batch_size: Optional[int] = 32
    # Number of workers for data loading. For heavier datasets, use less number of workers to prevent OOM errors
    num_workers: Optional[int] = 4
    # Manifold path to save checkpoints to
    manifold_path: Optional[str] = "tree/anjul/Supernet_qat"
    manifold_bucket: Optional[str] = "on_device_ai_cv_publicdata"
    # Whether to use GPU accelerator in pytorch lightning
    accelerator: Optional[str] = "gpu"
    # PL parameters
    strategy: Optional[str] = "ddp"
    # Number of nodes to use.
    num_node: Optional[int] = 1
    # Number of GPUs to use. Generally each node as 8 gpus
    num_gpu: Optional[int] = 1
    # Num CPUs to use for running the training job per node
    num_cpu: Optional[int] = 20
    # Memory each node in torch elastic is going to take. Generally 240g is max we can get per node on fbl
    memory: Optional[Union[str, int]] = -1
    # List of layers to attach qconfig to in a model. Use [''] to set qconfig for the full model
    # Append with 'module.' when using qat. For example features.0 becomes module.features.0 during qat
    # because we surround the model with QuantWrapper()
    layers: Optional[List[str]] = None
    # Weight and Activation bit width choices for each value of layers array
    # Use only a single valued lists if not using supernet for each element
    # When using supernet qat, we can define lists longer than len(1).
    # Supernet uses these values to search through the best working value for a given layer
    wt_bitwidths: Optional[List[List[int]]] = None
    ac_bitwidths: Optional[List[List[int]]] = None
    # One of methods defined in https://fburl.com/code/dmr0kcvg for setting qconfig
    qconfig_method: Optional[str] = "minmax_wperchannel"
    # Seed value for pytorch lightning. Used to produce pre-defined results with exact values as from a previous experiment
    seed: Optional[int] = 0
    # Lit Trainer definition
    max_epochs: Optional[int] = 1000
    # Lit Trainer definition
    max_steps: Optional[int] = -1
    # Optimizer params
    lr: Optional[float] = 0.1
    # Optimizer params
    momentum: Optional[float] = 0.9
    # Optimizer params
    weight_decay: Optional[float] = 1e-4
    # Step size for learning rate schedulers
    lr_step_size: Optional[int] = 30
    # lr gamma for learning rate schedulers
    lr_gamma: Optional[float] = 0.1
    # Train the model or not. Set to false if using pretrianed weights and don't need training (backprop)
    train: Optional[bool] = True
    # Whether to perform final evaluation after training is complete for 1 epoch
    final_eval: Optional[bool] = False
    # Number of subnets to sample during supernet qat. Set to None when not doing supernet training
    num_subnets: Optional[int] = 4
    # Path of checkpoint to load the state dict from. Set to None to start training from scratch
    checkpoint_path: Optional[str] = None
    # State Dict path if loading a state dict instead of a checkpoint file. Checkpoint file is used in pytorch
    # lightning, while state_dict path is used with pytorch load_state_dict method
    state_dict_path: Optional[str] = None
    # Lit Trainer definition
    limit_train_batches: Optional[Union[int, float, None]] = None
    # Lit Trainer definition
    limit_val_batches: Optional[Union[int, float, None]] = None
    # True if the experiment is qat. For regular training, set to False
    is_qat: Optional[bool] = False
    # Check validation after every n epochs
    check_val_every_n_epoch: Optional[int] = 1
    # Optimizer, one of [sgd, adam]
    optimizer: Optional[str] = "sgd"
    # Whether to use pre-trained model from torchvision.models. Set to true when doing PTQ
    pretrained: Optional[bool] = False
    # whether to use data in test mode
    # data_test_mode: Optional[bool] = False
    # test_data_size: Optional[int] = 1000
    # test_data_num_classes: Optional[int] = 10
    # test_data_image_size: Optional[Tuple[int, ...]] = (3, 224, 224)
    # Teacher model name to be used during distillation.
    teacher_model: Optional[str] = "resnet101"
    # Capabilities of the hardware to run the workflow on. See
    # https://www.internalfb.com/code/fbsource/[bfbe0412ba0f]/fbcode/on_device_ai/fast_nas/workflows/lsq_n2uq_mbv2_workflow.py?lines=48
    capabilities: Optional[List[str]] = None
    # Use teacher set to true whe using the teacher model distillation for training
    use_teacher: Optional[bool] = False
    # Experiment type to perform, one of "lsq", "n2uq", "fp", and "toqm"
    exp_type: Optional[str] = "fp"
    autoresume: Optional[bool] = True
    # How many batches to accumulate during training
    accumulate_grad_batches: Optional[int] = 1
    # No. of warup epochs
    warmup_steps: Optional[int] = None


def get_search_space_samples(search_space, num_subnets):
    r"""
    For each module in the search space, return a possible configuration, one per subnet.
    The configurations are generated based on the sandwich scheme (min, max, random)

    Args:
        search_space: Dictionary mapping modules to possible quantization/sparsity configurations
        num_subnets: Number of subnets to generate

    Return:
        subnet_search_space_config: Configuration for each module per subnet
    """

    subnet_search_space_config = []
    for idx in range(num_subnets):
        sample = {}
        for mod_name in search_space:
            sample[mod_name] = {}
            if idx == 0:
                for key in search_space[mod_name]:
                    sample[mod_name][key] = search_space[mod_name][key].max()
            elif idx == 1:
                for key in search_space[mod_name]:
                    sample[mod_name][key] = search_space[mod_name][key].min()
            else:
                for key in search_space[mod_name]:
                    sample[mod_name][key] = search_space[mod_name][key].random()
        subnet_search_space_config.append(sample)
    return subnet_search_space_config


class SupernetLitModule(object):
    r"""
    Template class for defining supernet ready lightning modules
    """

    def train_subnet(self, *args, **kwargs):
        pass

    def prepare_supernet(self, *args, **kwargs):
        pass

    def set_ss_samples(self, samples):
        # Search space samples
        self.ss_samples = samples

    def validate_subnet(self, *args, **kwargs):
        pass


# Define pl specific modules for subnets
class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        lr: float = 0.1,
        lr_step_size: int = 1,
        lr_gamma: float = 0.98,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
        optimizer: str = "sgd",
    ):
        super().__init__()
        self.lr: float = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.weight_decay: float = weight_decay

        self.model = model
        self.loss: nn.Module = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def forward(self, input):
        return self.model(input)

    def _step(self, batch, phase):
        features, true_labels = batch
        logits = self.model(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        return torch.mean(loss)

    def training_step(self, batch, *args, **kwargs):
        return self._step(batch, phase="train")

    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch, phase="val")

    def test_step(self, batch, *args, **kwargs):
        return self._step(batch, phase="test")

    def configure_optimizers(
        self,
    ):
        """Instantiates an optimizer for optimization."""
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]


class MobilenetV2Lit(pl.LightningModule):
    """
    mbv2 PL class for custom models (mbv2_n2uq.py and mobilenet_v2_lsq.py)
    """

    def __init__(
        self,
        model,
        lr: float = 0.1,
        lr_step_size: int = 1,
        lr_gamma: float = 0.98,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
        teacher: str = "resnet101",
        label_smooth: float = 0.1,
        optimizer: str = "adam",
        use_teacher: bool = True,
        accumulate_grad_batches=1,
        epochs=128,
        warmup_steps=None,
    ):
        super().__init__()
        self.lr: float = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.weight_decay: float = weight_decay
        # Setup teacher model
        if use_teacher:
            self.teacher = models.__dict__[teacher](pretrained=True)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        self.model = model
        self.loss: nn.Module = nn.CrossEntropyLoss()
        self.loss_smooth: nn.Module = CrossEntropyLabelSmooth(num_classes, label_smooth)
        self.loss_kd = KD_loss()

        all_parameters = model.parameters()
        self.weight_parameters = []
        self.alpha_parameters = []

        for pname, p in model.named_parameters():
            if p.ndimension() == 4 and "bias" not in pname:
                self.weight_parameters.append(p)
            if (
                "quan1.a" in pname
                or "quan2.a" in pname
                or "quan3.a" in pname
                or "scale" in pname
                or "start" in pname
            ):
                self.alpha_parameters.append(p)

        weight_parameters_id = list(map(id, self.weight_parameters))
        alpha_parameters_id = list(map(id, self.alpha_parameters))
        other_parameters1 = list(
            filter(lambda p: id(p) not in weight_parameters_id, all_parameters)
        )
        self.other_parameters = list(
            filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1)
        )

        self.metrics_loss: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_loss": MeanMetric(nan_strategy="error")
                for phase in ("train", "val", "test")
            }
        )
        self.metrics_acc: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_accuracy": Accuracy(num_classes=num_classes)
                for phase in ("train", "val", "test")
            }
        )
        self.optimizer = optimizer
        # self.save_hyperparameters()
        self.use_teacher = use_teacher
        self.accumulate_grad_batches = (
            accumulate_grad_batches if accumulate_grad_batches else 1
        )
        self.num_classes = num_classes
        self.epochs = epochs
        self.warmup_steps = warmup_steps if warmup_steps else -1

    # pyre-ignore[14]: torchscript doesn't support *args and **kwargs
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each training batch."""
        # print("inside training step")
        return self._step(batch, phase="train")

    def validation_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each validation batch."""
        return self._step(batch, phase="val")

    def test_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each test batch."""
        return self._step(batch, phase="test")

    def configure_optimizers(
        self,
    ):
        """Instantiates an optimizer for optimization."""
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {"params": self.other_parameters},
                    {
                        "params": self.weight_parameters,
                        "weight_decay": self.weight_decay,
                    },
                ],
                lr=self.lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                [
                    {"params": self.alpha_parameters, "lr": self.lr / 10},
                    {"params": self.other_parameters, "lr": self.lr},
                    {
                        "params": self.weight_parameters,
                        "weight_decay": self.weight_decay,
                        "lr": self.lr,
                    },
                ],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        # )
        def warmup(epoch):
            if epoch < self.warmup_steps:
                # warm up lr
                lr_scale = 0.1 ** (self.warmup_steps - epoch)
            else:
                # lr_scale = 0.95 ** epoch
                lr_scale = 1.0 - epoch / self.epochs
            return lr_scale

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup, last_epoch=-1
        )
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    def _step(self, batch, phase):
        # print("inside step")
        """Runs a forward pass to calculate loss and metrics."""
        images, target = batch
        if phase == "train":
            logits = self.model(images)
            if self.use_teacher:
                logits_teacher = self.teacher(images)
                loss = self.loss_kd(logits, logits_teacher)
            else:
                loss = self.loss_smooth(logits, target)
        else:
            logits = self.model(images)
            loss = self.loss(logits, target)
        self.metrics_loss[f"{phase}_loss"].update(value=loss)
        self.metrics_acc[f"{phase}_accuracy"].update(preds=logits, target=target)
        return torch.mean(loss)

    def training_epoch_end(self, outputs):
        # print('train epoch end')
        self.log("loss/train", self.metrics_loss["train_loss"].compute())
        self.log("accuracy/train", self.metrics_acc["train_accuracy"].compute())
        self.metrics_loss["train_loss"].reset()
        self.metrics_acc["train_accuracy"].reset()
        print(self.scheduler)

    def validation_epoch_end(self, outputs):
        # print('val epoch end')
        self.log("loss/val", self.metrics_loss["val_loss"].compute())
        self.log("accuracy/val", self.metrics_acc["val_accuracy"].compute())
        self.metrics_loss["val_loss"].reset()
        self.metrics_acc["val_accuracy"].reset()

    def test_epoch_end(self, outputs):
        self.log("loss/test", self.metrics_loss["test_loss"].compute())
        self.log("accuracy/test", self.metrics_acc["test_accuracy"].compute())
        self.metrics_loss["test_loss"].reset()
        self.metrics_acc["test_accuracy"].reset()


class MobilenetV2FP(pl.LightningModule):
    """
    MobilenetV2 PL class for FP training with custom defined mbv2.py
    """

    def __init__(
        self,
        model,
        lr: float = 0.1,
        lr_step_size: int = 1,
        lr_gamma: float = 0.98,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
        teacher: str = "resnet101",
        label_smooth: float = 0.1,
        optimizer: str = "adam",
        use_teacher: bool = False,
    ):
        super().__init__()
        self.lr: float = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.weight_decay: float = weight_decay
        # Setup teacher model
        if use_teacher:
            self.teacher = models.__dict__[teacher](pretrained=True)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        self.model = model
        self.loss: nn.Module = nn.CrossEntropyLoss()
        self.loss_smooth: nn.Module = CrossEntropyLabelSmooth(num_classes, label_smooth)
        self.loss_kd = KD_loss()

        all_parameters = model.parameters()
        self.weight_parameters = []

        for pname, p in model.named_parameters():
            if "fc" in pname or "conv1" in pname or "pwconv" in pname:
                self.weight_parameters.append(p)
        weight_parameters_id = list(map(id, self.weight_parameters))
        self.other_parameters = list(
            filter(lambda p: id(p) not in weight_parameters_id, all_parameters)
        )

        self.metrics_loss: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_loss": MeanMetric(nan_strategy="error")
                for phase in ("train", "val", "test")
            }
        )
        self.metrics_acc: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_accuracy": Accuracy(num_classes=num_classes)
                for phase in ("train", "val", "test")
            }
        )
        self.optimizer = optimizer
        self.save_hyperparameters()
        self.use_teacher = use_teacher

    # pyre-ignore[14]: torchscript doesn't support *args and **kwargs
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each training batch."""
        return self._step(batch, phase="train")

    def validation_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each validation batch."""
        return self._step(batch, phase="val")

    def test_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each test batch."""
        return self._step(batch, phase="test")

    def configure_optimizers(
        self,
    ):
        """Instantiates an optimizer for optimization."""
        optimizer = torch.optim.SGD(
            [
                {"params": self.other_parameters},
                {"params": self.weight_parameters, "weight_decay": self.weight_decay},
            ],
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    def _step(self, batch, phase):
        """Runs a forward pass to calculate loss and metrics."""
        images, target = batch
        if phase == "train":
            logits = self.model(images)
            if self.use_teacher:
                logits_teacher = self.teacher(images)
                loss = self.loss_kd(logits, logits_teacher)
            else:
                loss = self.loss_smooth(logits, target)
        else:
            logits = self.model(images)
            loss = self.loss(logits, target)
        self.metrics_loss[f"{phase}_loss"].update(value=loss)
        self.metrics_acc[f"{phase}_accuracy"].update(preds=logits, target=target)
        return torch.mean(loss)

    def training_epoch_end(self, outputs):
        self.log("loss/train", self.metrics_loss["train_loss"].compute())
        self.log("accuracy/train", self.metrics_acc["train_accuracy"].compute())
        self.metrics_loss["train_loss"].reset()
        self.metrics_acc["train_accuracy"].reset()

    def validation_epoch_end(self, outputs):
        self.log("loss/val", self.metrics_loss["val_loss"].compute())
        self.log("accuracy/val", self.metrics_acc["val_accuracy"].compute())
        self.metrics_loss["val_loss"].reset()
        self.metrics_acc["val_accuracy"].reset()

    def test_epoch_end(self, outputs):
        self.log("loss/test", self.metrics_loss["test_loss"].compute())
        self.log("accuracy/test", self.metrics_acc["test_accuracy"].compute())
        self.metrics_loss["test_loss"].reset()
        self.metrics_acc["test_accuracy"].reset()


def create_search_space(search_space_dict):
    r"""
    Maps the search space parameters to fastnas search space classes.

    Args:
        search_space_dict: Dictionary specifying quantization and/or sparsity qconfigs
        e.g. {"quantization": {"features.0": [(4, 8), (8, 8), (8, 16)]}, "sparsity": {"features.0": (0.1, 0.9)}}
        where 'features.0' is the module name in the model which is to be quantized for given settings.

    Return:
        search_space: Search Space params mapped to the fastnas search space classes
    """
    search_space = {}
    if "quantization" in search_space_dict.keys():
        for mod_name in search_space_dict["quantization"]:
            search_space[mod_name] = {}
            search_space[mod_name]["quantization"] = Choice(
                *search_space_dict["quantization"][mod_name]
            )

    if "sparsity" in search_space_dict.keys():
        for mod_name in search_space_dict["sparsity"]:
            search_space[mod_name]["sparsity"] = Uniform(
                search_space_dict["sparsity"][mod_name][0],
                search_space_dict["sparsity"][mod_name][1],
                reverse=True,
            )
    return search_space


class SupernetTraining(Callback):
    def __init__(self, search_space_dict, num_subnets):
        self.num_subnets = num_subnets
        self.search_space_dict = search_space_dict
        self.search_space = create_search_space(search_space_dict)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Provide a list of all the ss_samples to use for this batch
        ss_sample_dict = get_search_space_samples(self.search_space, self.num_subnets)
        pl_module.set_ss_samples(ss_sample_dict)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        ss_sample_dict = get_search_space_samples(self.search_space, 2)
        pl_module.set_ss_samples(ss_sample_dict)


# Define pl specific modules for supernet
class SupernetLitModel(MobilenetV2Lit, SupernetLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_loss: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_loss": MeanMetric(nan_strategy="error")
                for phase in ("train", "val_max", "val_min", "test_max", "test_min")
            }
        )
        self.metrics_acc: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_accuracy": Accuracy(num_classes=self.num_classes)
                for phase in ("train", "val_max", "val_min", "test_max", "test_min")
            }
        )

    # def prepare_supernet(self, ss_config, ss_dict):
    #     # function that creates a supernet ready model.
    #     prepare_supernet(self.model, ss_config, ss_dict)

    def set_active_filter(self, samp):
        activate_subnet(self.model, samp)

    # Required API in lightning module
    def train_subnet(self, batch, batch_idx, loss_mod):
        return self._step(batch, phase="train")

    def validate_subnet(self, batch, batch_idx, loss_mod, subnet_type):
        return self._step(batch, phase="val", subnet_type=subnet_type)

    def training_epoch_end(self, outputs):
        # print('train epoch end')
        self.log("loss/train", self.metrics_loss["train_loss"].compute())
        self.log("accuracy/train", self.metrics_acc["train_accuracy"].compute())
        self.metrics_loss["train_loss"].reset()
        self.metrics_acc["train_accuracy"].reset()
        self.scheduler.step()

    def validation_epoch_end(self, outputs):
        # print('val epoch end')
        self.log("loss/max/val", self.metrics_loss["val_max_loss"].compute())
        self.log("loss/min/val", self.metrics_loss["val_min_loss"].compute())
        self.log("accuracy/max/val", self.metrics_acc["val_max_accuracy"].compute())
        self.log("accuracy/min/val", self.metrics_acc["val_min_accuracy"].compute())
        self.metrics_loss["val_max_loss"].reset()
        self.metrics_loss["val_min_loss"].reset()
        self.metrics_acc["val_max_accuracy"].reset()
        self.metrics_acc["val_min_accuracy"].reset()

    def test_epoch_end(self, outputs):
        # print('val epoch end')
        self.log("loss/max/test", self.metrics_loss["test_max_loss"].compute())
        self.log("loss/min/test", self.metrics_loss["test_min_loss"].compute())
        self.log("accuracy/max/test", self.metrics_acc["test_max_accuracy"].compute())
        self.log("accuracy/min/test", self.metrics_acc["test_min_accuracy"].compute())
        self.metrics_loss["test_max_loss"].reset()
        self.metrics_loss["test_min_loss"].reset()
        self.metrics_acc["test_max_accuracy"].reset()
        self.metrics_acc["test_min_accuracy"].reset()

    def _step(self, batch, phase, subnet_type=None):
        # print("inside step")
        """Runs a forward pass to calculate loss and metrics."""
        images, target = batch
        if phase == "train":
            logits = self.model(images)
            if self.use_teacher:
                logits_teacher = self.teacher(images)
                loss = self.loss_kd(logits, logits_teacher)
            else:
                loss = self.loss_smooth(logits, target)
            self.metrics_loss[f"{phase}_loss"].update(value=loss)
            self.metrics_acc[f"{phase}_accuracy"].update(preds=logits, target=target)
        else:
            logits = self.model(images)
            loss = self.loss(logits, target)
            self.metrics_loss[f"{phase}_{subnet_type}_loss"].update(value=loss)
            self.metrics_acc[f"{phase}_{subnet_type}_accuracy"].update(
                preds=logits, target=target
            )
        return torch.mean(loss)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_batch_size, val_batch_size, num_workers):
        super().__init__()
        self.data_path = data_path
        self.trainloader, self.testloader = None, None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_loader, self.val_loader, _ = build_imagenet_dataloader(
            batch_size=self.train_batch_size,
            val_batch_size=self.val_batch_size,
            ddp_sampler=True,
            ddp_val_sampler=True,
            num_workers=self.num_workers,
        )

        return

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader


class DataModuleTest(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        size=100000,
        num_classes=10000,
        image_size=(3, 224, 224),
    ) -> None:
        """
        Initializes Fake dataset for training, validation, and testing.

        Args:
            batch_size: the batch size each dataloader uses.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dataset = torchvision.datasets.FakeData(
            size=size,
            num_classes=num_classes,
            image_size=image_size,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )

        self.datasets = {}
        self.datasets["train"] = dataset
        self.datasets["val"] = dataset
        self.datasets["test"] = dataset

    def train_dataloader(self):
        """Returns a dataloader for training."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """Returns a dataloader for validation."""
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        """Returns a dataloader for testing."""
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class PLModel(pl.LightningModule):
    """
    PL class to train torchvision models from scratch. Tested on Mobilenet
    """

    def __init__(
        self,
        model,
        lr: float = 0.1,
        lr_step_size: int = 1,
        lr_gamma: float = 0.98,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
        optimizer: str = "sgd",
    ):
        super().__init__()
        self.lr: float = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.weight_decay: float = weight_decay

        self.model = model
        self.loss: nn.Module = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.metrics_loss: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_loss": MeanMetric(nan_strategy="error")
                for phase in ("train", "val", "test")
            }
        )
        self.metrics_acc: nn.ModuleDict = nn.ModuleDict(
            {
                f"{phase}_accuracy": Accuracy(num_classes=num_classes)
                for phase in ("train", "val", "test")
            }
        )

        # TODO: fix this when using with qat
        # self.save_hyperparameters()

    # pyre-ignore[14]: torchscript doesn't support *args and **kwargs
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each training batch."""
        return self._step(batch, phase="train")

    def validation_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each validation batch."""
        return self._step(batch, phase="val")

    def test_step(self, batch, *args, **kwargs):
        """Defines logic to execute on each test batch."""
        return self._step(batch, phase="test")

    def configure_optimizers(
        self,
    ):
        """Instantiates an optimizer for optimization."""
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    def _step(self, batch, phase):
        """Runs a forward pass to calculate loss and metrics."""
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        self.metrics_loss[f"{phase}_loss"].update(value=loss)
        self.metrics_acc[f"{phase}_accuracy"].update(preds=output, target=target)
        return torch.mean(loss)

    def training_epoch_end(self, outputs):
        self.log("loss/train", self.metrics_loss["train_loss"].compute())
        self.log("accuracy/train", self.metrics_acc["train_accuracy"].compute())
        self.metrics_loss["train_loss"].reset()
        self.metrics_acc["train_accuracy"].reset()

    def validation_epoch_end(self, outputs):
        self.log("loss/val", self.metrics_loss["val_loss"].compute())
        self.log("accuracy/val", self.metrics_acc["val_accuracy"].compute())
        self.metrics_loss["val_loss"].reset()
        self.metrics_acc["val_accuracy"].reset()

    def test_epoch_end(self, outputs):
        self.log("loss/test", self.metrics_loss["test_loss"].compute())
        self.log("accuracy/test", self.metrics_acc["test_accuracy"].compute())
        self.metrics_loss["test_loss"].reset()
        self.metrics_acc["test_accuracy"].reset()


class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, "The bit-width of identity quantizer must be None"

    def forward(self, x):
        return x


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2**bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1)) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1))
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = torch.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)
                * 2
                / (self.thd_pos**0.5)
            )
        else:
            self.s = torch.nn.Parameter(
                x.detach().abs().mean() * 2 / (self.thd_pos**0.5)
            )

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class KD_loss(nn.modules.loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.
    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss


def key_transformation(key_name):
    return key_name.replace("module.", "")


def rename_state_dict_keys(source, key_transformation, target=None):
    state_dict = source
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def activate_subnet(model, ss_sample_dict):
    r"""
    Given a current search space sample for a module, set the configuration in place

    Args:
        model: Prepared pytorch model - from fast_nas_prepare_supernet
        ss_sample_dict: Dictionary mapping a module to the chosen search space sample
        e.g. {'model.layer1': {'quantization': (4, 8)}}
    """
    for mod_name in ss_sample_dict:
        mod = fqn_to_module(model, mod_name)
        if mod and type(mod) in [HardQuantizeConv, lsqquan_toqm, quan_bn]:
            mod.set_active_filter(ss_sample_dict[mod_name])


def create_module(module_class, loss_modules, *args, **kwargs):
    r"""
    Converts a lightning model to a supernet model and defines a training step.
    """

    class Module(module_class):
        def __init__(self, loss_modules, *args, **kwargs):
            module_class.__init__(self, *args, **kwargs)
            # Prepare model for supernet training here or can optionally be on train_start
            self.automatic_optimization = False
            self.loss_module = loss_modules

        def training_step(self, batch, batch_idx):
            # print("inside supernet training step")
            opt = self.optimizers()
            # for each ss sample, we change the module for that sample
            for samp in self.ss_samples:
                self.set_active_filter(
                    samp
                )  # Sets the bitwidths of every layer in the network
                loss = self.train_subnet(batch, batch_idx, self.loss_module)
                self.log("training_step_loss", loss, prog_bar=True)
                # do a manual backward pass instead of lit
                self.manual_backward(loss)

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt.step()
                opt.zero_grad()
            return loss

        def validation_step(self, batch, batch_idx):
            for i, samp in enumerate(self.ss_samples):
                # print(f"Current Sample: {samp}")
                self.set_active_filter(
                    samp
                )  # Sets the bitwidths of every layer in the network
                subnet_type = "max" if i % 2 == 0 else "min"
                loss = self.validate_subnet(
                    batch, batch_idx, self.loss_module, subnet_type
                )
            return loss

    return Module(loss_modules, *args, **kwargs)


class Item:
    def __init__(self, filename: str, entry: DirectoryEntry, directory: str):
        split_filename = os.path.splitext(filename)
        self.name = split_filename[0]
        self.file_type = split_filename[1][1:]

        self.ctime = entry.meta.ctime
        self.properties = dict(entry.meta.properties)
        self.size = entry.blobSizeBytes
        self.directory = directory

    def __str__(self) -> str:
        return f"name='{self.name}' file_type='{self.file_type}' properties='{self.properties}' ctime={self.ctime} size={self.size}"


async def async_list_directory(client, manipath) -> Tuple[List[Item], List[str]]:
    files = []
    directories = []
    async for filename, meta in client.ls(manipath):
        if meta.blobSizeBytes == 0 and meta.entryType is not DirectoryEntryType.BLOB:
            directories.append(os.path.join(manipath, filename))
        else:
            files.append(Item(filename, meta, manipath))

    return files, directories


async def async_recursive_ls(bucket: str, path: str, max_qps: int = 300):
    output_files = []
    queue = deque()
    queue.append([path])
    with ManifoldClient.get_client(bucket) as client:
        while queue:
            backlog = queue.pop()
            tasks = [
                asyncio.create_task(async_list_directory(client, directory))
                for directory in backlog
            ]
            r = await asyncio.gather(*tasks)
            next_directories = []
            for files, directories in r:
                output_files.append(files)
                next_directories.append(directories)
            # Next level
            next_directories = list(itertools.chain.from_iterable(next_directories))
            for i in range(0, len(next_directories), max_qps):
                next_backlog = next_directories[i : i + max_qps]
                queue.appendleft(next_backlog)
    return [items for file_list in output_files for items in file_list]
