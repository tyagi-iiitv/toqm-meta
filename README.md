# Train Once and Quantize Many (TOQM)

## Launch Command

- For on-demand GPU
```
FLOW_BUCK_OPTS="-c fbcode.enable_gpu_sections=true" flow-cli test-locally --canary-driver --mode opt fblearner.flow.projects.users.anjultyagi.lsq_n2uq_toqm_mbv2_workflow.workflow@//fblearner/flow/projects/users/anjultyagi:workflow --parameters-file fblearner/flow/projects/users/anjultyagi/configs/toqm_ps.json --buck-version v2
```
- For lanching an fbl experiment
```
FLOW_BUCK_OPTS="-c fbcode.enable_gpu_sections=true" flow-cli canary fblearner.flow.projects.users.anjultyagi.lsq_n2uq_toqm_mbv2_workflow.workflow@//fblearner/flow/projects/users/anjultyagi:workflow  --run-as-secure-group team_fast_ai_team --entitlement default_ftw_gpu --mode=opt --name "supernet qat training" --parameters-file fblearner/flow/projects/users/anjultyagi/configs/toqm_full_config.json --buck-version v2
```

Use one of the following entitlements
- default_ftw_gpu
- bigbasin_atn_arvr

## Configuration Files
The config json files are present in the `configs/*` directory. For toqm experiments, please use either of `configs/toqm_full_config.json` for full scale experiments, or `configs/toqm_ps.json` for small experiments to test the code.

## Results File
This is an ongoing experiment, please update the results in the following sheet in case you run any experiments.

[TOQM Results Sheet](https://docs.google.com/spreadsheets/d/1dsMkUn-Inm3BYeRdCMoNO7jZI3FgZ3kqBNQchjyLkcM/edit?usp=sharing)

## Code Repo Github
This project was started as an intern project and is now modified outside of Meta. Please access the following github link for latest code updates.

[tyagi-iiitv/toqm-meta](https://github.com/tyagi-iiitv/toqm-meta). You can request access if you need by sending an email to aktyagi@cs.stonybrook.edu

## Configuration Parameters
```
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
data_test_mode: Optional[bool] = False
test_data_size: Optional[int] = 1000
test_data_num_classes: Optional[int] = 10
test_data_image_size: Optional[Tuple[int, ...]] = (3, 224, 224)

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
```

## Follow the progress
- Task T131374631
