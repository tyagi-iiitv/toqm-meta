#!/usr/bin/env python3

import os

import fblearner.flow.api as flow

import torch
import torch.distributed.launcher.fb.flow_launch as launch
from fblearner.flow.api import runtimecontext
from fblearner.flow.projects.users.anjultyagi.mbv2 import MobileNetV2 as MobileNetV2_fp

from fblearner.flow.projects.users.anjultyagi.mbv2_n2uq import (
    MobileNetV2 as MobileNetV2_n2uq,
)
from fblearner.flow.projects.users.anjultyagi.mbv2_toqm import (
    MobileNetV2 as MobileNetV2_toqm,
)
from fblearner.flow.projects.users.anjultyagi.mobilenet_v2_lsq import (
    MobileNetV2 as MobileNetV2_lsq,
)

from fblearner.flow.projects.users.anjultyagi.utils import (
    async_recursive_ls,
    create_module,
    DataModule,
    # DataModuleTest,
    key_transformation,
    MobilenetV2FP,
    MobilenetV2Lit,
    Params,
    rename_state_dict_keys,
    SupernetLitModel,
    SupernetTraining,
)
from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from libfb.py.asyncio.await_utils import await_sync
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from stl.lightning.callbacks.model_checkpoint import ModelCheckpoint
from stl.lightning.loggers.manifold_tensorboard_logger import ManifoldTensorBoardLogger
from torchtnt.utils import init_from_env


@flow.registered(owners=["oncall+on_device_ai"])
@flow.typed()
def workflow(params: Params):
    elastic_parameters = launch.LaunchConfig(
        min_nodes=params.num_node,
        max_nodes=params.num_node,
        nproc_per_node=params.num_gpu if params.accelerator != "cpu" else 1,
    )
    resource_requirements = flow.ResourceRequirements(
        cpu=params.num_cpu,
        gpu=params.num_gpu,
        memory=params.memory,
        capabilities=params.capabilities,
    )
    workflow_id = runtimecontext.get_flow_environ().workflow_run_id
    ret = launch.elastic_launch(elastic_parameters, train_op)(
        params, workflow_id, resource_requirements=resource_requirements
    )
    return ret[0]


def train_op(params: Params, workflow_id: int):  # noqa
    print("Saving files under: {}".format(workflow_id))
    device = init_from_env()
    default_accelerator = "gpu" if device.type == "cuda" else "cpu"
    if default_accelerator != params.accelerator and params.accelerator != "auto":
        print(
            "the default accelerator {} is different from the specified accelerator {}".format(
                default_accelerator, params.accelerator
            )
        )

    # Pin memory for reproducible results, required when working with multiple GPUs
    pin_memory = True if params.accelerator == "gpu" else False
    seed_everything(params.seed, workers=True)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", 1))
    devices = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    print("World size: {}".format(world_size))
    print("Number of nodes: {}".format(num_nodes))
    print("Number of devices: {}".format(devices))
    print("Number of GPUs: {}".format(torch.cuda.device_count()))
    print("Pin memory: {}".format(pin_memory))

    if params.batch_size % world_size != 0:
        raise RuntimeError(
            "batch_size={}, nodes={}, devices={}, and world_size={}, the batch_size should be an integer multiple of world_size".format(
                params.batch_size, num_nodes, devices, world_size
            )
        )
    batch_size_per_device = params.batch_size // world_size
    print("batch_size_per_device: {}".format(batch_size_per_device))

    data_module = DataModule(
        os.getcwd(),
        batch_size_per_device,
        batch_size_per_device,
        params.num_workers,
    )
    data_module.prepare_data()
    data_module.setup()

    models = {
        "lsq": MobileNetV2_lsq(params.wt_bitwidths[0][0], params.ac_bitwidths[0][0]),
        "n2uq": MobileNetV2_n2uq(params.wt_bitwidths[0][0], params.ac_bitwidths[0][0]),
        "toqm": MobileNetV2_toqm(params.wt_bitwidths[0][0], params.ac_bitwidths[0][0]),
        "fp": MobileNetV2_fp(),
    }

    model = models[params.exp_type]

    if params.state_dict_path:
        try:
            path_manager = PathManager()
            path_manager.register_handler(ManifoldPathHandler())
        except BaseException as e:
            print(e)
            pass
        with path_manager.open(params.state_dict_path, "rb") as f:
            model_zip = torch.load(f)
            state_dict = rename_state_dict_keys(
                model_zip["state_dict"], key_transformation
            )
            model.load_state_dict(state_dict, strict=False)

    grad_acc = params.accumulate_grad_batches
    if params.exp_type in ["lsq", "n2uq"]:
        module = MobilenetV2Lit(
            model=model,
            lr=params.lr,
            lr_step_size=params.lr_step_size,
            lr_gamma=params.lr_gamma,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
            use_teacher=params.use_teacher,
            epochs=params.max_epochs,
            warmup_steps=params.warmup_steps,
        )
    elif params.exp_type == "toqm":
        module = create_module(
            SupernetLitModel,
            [],
            model=model,
            lr=params.lr,
            lr_step_size=params.lr_step_size,
            lr_gamma=params.lr_gamma,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
            use_teacher=params.use_teacher,
            accumulate_grad_batches=grad_acc,
            epochs=params.max_epochs,
            warmup_steps=params.warmup_steps,
        )
        grad_acc = None
    else:
        module = MobilenetV2FP(
            model=model,
            lr=params.lr,
            lr_step_size=params.lr_step_size,
            lr_gamma=params.lr_gamma,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
            use_teacher=params.use_teacher,
        )

    # Instantiate trainer
    logger = ManifoldTensorBoardLogger(
        manifold_bucket=params.manifold_bucket,
        manifold_path=params.manifold_path,
        has_user_data=False,
        ttl_days=365,
        name=str(workflow_id),
    )

    checkpoint_callback = ModelCheckpoint(
        has_user_data=False, ttl_days=365, every_n_val_epochs=1
    )

    search_space_vals = {}
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(),
    ]
    if params.exp_type == "toqm":
        if params.layers != [""]:
            for i, layer in enumerate(params.layers):
                cur_vals = []
                for val in zip(params.wt_bitwidths[i], params.ac_bitwidths[i]):
                    cur_vals.append(val)
                search_space_vals[layer] = cur_vals
        else:
            cur_vals = []
            for val in params.wt_bitwidths[0]:
                for val2 in params.ac_bitwidths[0]:
                    cur_vals.append((val, val2))
            for layer, _ in model.named_modules():
                search_space_vals[layer] = cur_vals

        search_space_dict = {"quantization": search_space_vals}

        supernet_callback = SupernetTraining(search_space_dict, params.num_subnets)
        callbacks.append(supernet_callback)

    trainer = Trainer(
        max_epochs=params.max_epochs,
        max_steps=params.max_steps,
        num_nodes=num_nodes,
        devices=devices,
        accelerator=params.accelerator,
        strategy=params.strategy,
        logger=logger,
        callbacks=callbacks,
        limit_train_batches=params.limit_train_batches,
        limit_val_batches=params.limit_val_batches,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        accumulate_grad_batches=grad_acc,
    )

    ckpt_path = params.checkpoint_path

    if params.autoresume:
        try:
            files = await_sync(
                async_recursive_ls(
                    params.manifold_bucket,
                    params.manifold_path + "/" + str(workflow_id),
                )
            )
            ckpt_path = sorted(
                [
                    "manifold://"
                    + params.manifold_bucket
                    + "/"
                    + file.directory
                    + "/"
                    + file.name
                    + "."
                    + file.file_type
                    for file in files
                    if file.name == "last"
                ]
            )[-1]
        except Exception as e:
            print(e)
            print("Training from scratch now")
            ckpt_path = None

    # Model fitting
    if params.train:
        try:
            trainer.fit(
                model=module,
                datamodule=data_module,
                ckpt_path=ckpt_path,
            )
        except Exception:
            print("loading from checkpoint failed, starting a fresh training job")
            trainer.fit(
                model=module,
                datamodule=data_module,
                ckpt_path=None,
            )

    outputs = {
        "best_model_path": checkpoint_callback.best_model_path,
        "tensorboard_log_dir": logger.save_dir,
        "last_model_path": checkpoint_callback.last_model_path,
    }

    # Evaluation
    if params.final_eval:
        last_val_estimates = trainer.validate(module, data_module.val_dataloader())
        outputs["last_val_estimates"] = last_val_estimates[0]

    return outputs
