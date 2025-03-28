# Gen5 NAS

## Overview

Gen5 NAS is an optimal network exploration tool for R-Car Gen5. It has the following features:

* HW-based NAS execution is supported
* Neural network architecture can be learned with a single network
* Training is not required during exploration
* The best architectures can be visualized using Pareto front graphs

It supports the following tasks:

* Classification
* Object Detection

HW cost acquisition uses PPAEstimator via RPC.

## Environment

The tool has been tested in the following environment:

* Ubuntu 18.04 LTS
* CPU: Intel(R) Xeon(R) Gold 6152 CPU @ 2.10GHz x2
* RAM: 512GB
* NVIDIA Tesla V100 (32GB VRAM) x 8
* NVIDIA driver 550.107.02
* CUDA 11.3
* Docker 20.10.16

## Quick Start

This chapter explains the procedure for executing NAS using a predefined configuration file.

### Setup

This section will explain how to setup the required environment to run the NAS tool.

#### Data Preparation

Prepare the dataset.

In this NAS, the following datasets have been confirmed for each task:

* Classification Task: ImageNet
* Object Detection Task: COCO

#### Docker Environment Startup

This program is designed to run on a Docker container, so creating a Docker image and container is necessary. In addition, to use NVIDIA GPU with Docker, it is required to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host system.

* Create a Docker image:

    ```
    $ cd ${WORK_DIR}
    $ ./docker/mmrazor/docker_build.sh
    ```

* Launch the Docker container:

    ```
    $ cd ${WORK_DIR}
    $ HOST_IMAGENET_PATH="/host/imagenet/path" HOST_COCO_PATH="/host/coco/path" ./docker/mmrazor/docker_run.sh
    ```

    * Set `HOST_IMAGENET_PATH` to the path of ImageNet on the host PC. It will be mounted in the container.
    * Set `HOST_COCO_PATH` to the path of COCO on the host PC. It will be mounted in the container.

Execute NAS on the generated container for subsequent operations.

#### PPAEstimator Server Preparation

1.  PPAEstimator's Docker environment startup (use a separate terminal from NAS):

    * Create a Docker image:

        ```
        $ cd ${WORK_DIR}/modules/ppa-estimator
        $ ./docker/build.sh
        ```

    * Start the Docker container:

        ```
        $ cd ${WORK_DIR}/modules/ppa-estimator
        $ ./docker/run.sh
        ```

    * Install additional modules for RPC:

        ```
        $ cd ${WORK_DIR}/modules/ppa-estimator
        $ ./internal/tools/grpc/check_and_install_for_server.sh
        ```

2.  Start the PPAEstimator Server:

    ```
    $ cd ${WORK_DIR}/modules/ppa-estimator
    $ python internal/tools/grpc/server.py
    ```

    * The server port is 50051 by default. (Can be specified with the `--port` option)
    * The PPAEstimator Server used in the search phase of this NAS can be used simultaneously with multiple instances.
    * Multiple ports can be used to enable parallel process execution.

    ```
    $ cd ${WORK_DIR}/modules/ppa-estimator
    $ python internal/tools/grpc/server.py --port 50051 50052 50053 50054
    ```

### NAS Execution

NAS execution is divided into two steps:

* Supernet(A neural network architecture search space represented as a single network with shared weights) training step
* Subnet(A single network extracted from the Supernet) exploration step

The configuration files for training and exploration are stored in `./configs/nas/`.

* Classification Task: `./mmcls`
* Object Detection Task: `./mmdet`

The configuration files for each step are different. The correspondence is as follows:

* `*_supernet*.py`: Training step
* `*_search*.py`: Exploration step
* `*_subnet*.py`: Re-Training step

In the following example, we will use the following configuration files:

* Training configuration file (training step): `configs/nas/mmcls/compbignas/mobilenetv2_sample1_supernet_in1k.py`
* Exploration configuration file (exploration step): `configs/nas/mmcls/compbignas/mobilenetv2_sample1_search_in1k.py`
* Re-Training configuration file (retraining step): `configs/nas/mmcls/compbignas/mobilenetv2_sample1_subnet_in1k.py`

**Executing the Training Step**

The front-end script for the training step is `./modules/mmrazor/tools/dist_train.sh`.
You can execute NAS (Neural Architecture Search) training using the following command:

```
$ CUDA_VISIBLE_DEVICES="0,1,2,3" PORT="29504" bash ./modules/mmrazor/tools/dist_train.sh \
    configs/nas/mmcls/compbignas/mobilenetv2_sample1_supernet_in1k.py \
    4 \
    --work-dir "data/local/nas_run_train_cls_test"
```

Below is an explanation of the arguments and environment variables to the script:

- **Environment Variable `CUDA_VISIBLE_DEVICES`**: Specifies the GPU indices to use.
- **Environment Variable `PORT`**: Specifies the port number for inter-process communication required by NAS. Make sure the port is available.
- **First Argument**: Path to the configuration file for NAS training.
- **Second Argument**: Number of GPUs to use. This should match the total number of indices specified in `CUDA_VISIBLE_DEVICES`.
- **`--work-dir`**: Path to the directory where the results of the NAS training will be saved.

After executing the command, the output directory will be populated with the following data:

```
data/local/nas_run_train_cls_test/
|-- 20250211_094023/ : Contains logs and other execution details
|-- best_max_subnet.accuracy_top1_epoch_1.pth : Checkpoint file of the model with the best score (used during the search step)
|-- epoch_1.pth : Checkpoint file saved after each epoch
|-- last_checkpoint : Symbolic link to the latest checkpoint file
`-- mobilenetv2_sample1_supernet_in1k.py : A copy of the configuration file used during training
```

**Executing the Search Step**

The front-end script for the search step is also `./modules/mmrazor/tools/dist_train.sh`.
You can execute the NAS search using the following command:

```
$ CUDA_VISIBLE_DEVICES="0" PORT="29505" bash ./modules/mmrazor/tools/dist_train.sh \
    configs/nas/mmcls/compbignas/mobilenetv2_sample1_search_in1k.py \
    1 \
    --work-dir "data/local/nas_run_search_cls_test" \
    --cfg-options load_from="data/local/nas_run_train_cls_test/best_max_subnet.accuracy_top1_epoch_360.pth"
```

Below is an explanation of the arguments and environment variables to the script:

- **Environment Variable `CUDA_VISIBLE_DEVICES`**: Specifies the GPU index to use. Only one GPU is allowed during the search step.
- **Environment Variable `PORT`**: Specifies the port number for inter-process communication required by NAS. Make sure the port is available.
- **First Argument**: Path to the configuration file for the NAS search.
- **Second Argument**: Number of GPUs to use. This should match the number of indices specified in `CUDA_VISIBLE_DEVICES`.
- **`--work-dir`**: Path to the directory where the results of the NAS search will be saved.
- **`--cfg-options`**: Path to the checkpoint file generated during the training step. Specify using the format `load_from=""`.

After executing the command, the output directory will be populated with the following data:

```
data/local/nas_run_search_cls_test/
|-- 20250211_112458/ : Contains logs and other execution details
|-- fix_subnets/ : Contains configuration files for the model architecture from each search iteration
|   |-- trial_0.yaml
|   |-- trial_1.yaml
|   `-- ...
|-- mobilenetv2_sample1_search_in1k.py : A copy of the configuration file used during the search
|-- pareto_front.html : Graph visualization search results (viewable in a web browser)
|-- pareto_front.png : Image version of the `pareto_front.html` results
|-- study.pkl : Intermediate results from the search process
`-- subnets.csv : Numerical results from the search, saved in CSV format
```
**Executing the Re-Training Step**

The front-end script for the retraining step is `./modules/mmrazor/tools/dist_train.sh`.
You can execute NAS (Neural Architecture Search) retraining using the following command:

```
$ CUDA_VISIBLE_DEVICES="0,1,2,3" PORT="29504" bash ./modules/mmrazor/tools/dist_train.sh \
    configs/nas/mmcls/compbignas/mobilenetv2_sample1_subnet_in1k.py \
    4 \
    --work-dir "data/local/nas_run_re_train_cls_test"
```
Below is an explanation of the arguments and environment variables to the script:

- **Environment Variable `CUDA_VISIBLE_DEVICES`**: Specifies the GPU indices to use.
- **Environment Variable `PORT`**: Specifies the port number for inter-process communication required by NAS. Make sure the port is available.
- **First Argument**: Path to the configuration file for NAS retraining.
- **Second Argument**: Number of GPUs to use. This should match the total number of indices specified in `CUDA_VISIBLE_DEVICES`.
- **`--work-dir`**: Path to the directory where the results of the NAS retraining will be saved.

After executing the command, the output directory will be populated with the following data:

```
data/local/nas_run_re_train_cls_test/
|-- 20250301_090704/ : Contains logs and other execution details
`-- mobilenetv2_sample1_subnet_in1k.py : A copy of the configuration file used during retraining
```

**(Optional) Exporting the Model to ONNX Format**

The front-end script for converting the model to ONNX is `./modules/mmrazor/tools/model_converters/convert_subnet2onnx.py`.
You can perform the ONNX conversion using the following command:

```
$ python ./modules/mmrazor/tools/model_converters/convert_subnet2onnx.py \
    configs/nas/mmcls/compbignas/mobilenetv2_sample1_subnet_in1k.py \
    data/local/nas_run_train_cls_test/best_max_subnet.accuracy_top1_epoch_360.pth \
    data/local/nas_run_search_test/fix_subnets/trial_2.yaml \
    1,3,224,224 \
    --work-dir data/local/nas_run_onnx \
    --output data/local/nas_run_onnx/trial_2.onnx
```

Below is an explanation of the arguments and environment variables to the script:

- **First Argument**: Path to the configuration file used during ONNX conversion:
    - **Classification Task:** `configs/nas/mmcls/compbignas/mobilenetv2_sample1_subnet_in1k.py`
    - **Object Detection Task:** `configs/nas/mmdet/compbignas/mobilenetv2_det_sample1_subnet.py`
- **Second Argument**: Path to the checkpoint file generated during the training step.
- **Third Argument**: Path to the architecture configuration file generated during the search step.
- **Fourth Argument**: Specifies the input resolution for the model.
- **`--work-dir`**: Path to the directory where logs will be saved.
- **`--output`**: Path to save the output ONNX file.

After executing the command, the output directory will be populated with the following data:

```
data/local/nas_run_onnx/
|-- 20250211_160600/ : Contains logs and other execution details
|-- mobilenetv2_sample1_subnet_in1k.py : A copy of the configuration file used during ONNX conversion
`-- trial_2.onnx : The converted ONNX model file
```

## Specifications

This chapter explains the specifications of NAS.

### Common NAS Specifications

Gen5 NAS searches for architectures on the Pareto front, considering the trade-off between hardware cost and accuracy on the R-Car Gen5 platform.

NAS is executed through the following steps:

1. Training step (SuperNet training)
2. Search step (Subnet search)

Details will be explained in the following sections.

The following are provided as inputs:

- Training configuration file
- Search configuration file

The following outputs are generated:

- Trained SuperNet model
- List of Pareto front architectures
- Trained models of Pareto front architectures

#### Training Step

This section will explain how to train the SuperNet.

**About the Algorithm**

Unlike standard neural network training, a single iteration involves four forward passes with switched node connections. This approach helps prevent accuracy degradation during the search step.

The training method is based on the algorithm from the paper [BigNAS](https://arxiv.org/abs/2003.11142). It incorporates elements from [CompOFA](https://arxiv.org/abs/2104.12642) to overcome the major drawback of BigNAS: an excessively large search space.

**Search Targets**

For both classification and object detection tasks, only the backbone is targeted for search.

The parameters explored are as follows:

- Global Parameters:
  - `input_resizer_cfg`: Input size
  - `first_num_out_channels`: Number of output channels for the first convolution
  - `last_num_out_channels`: Number of output channels for the backbone

- Stage Parameters:
  - `num_blocks`: Number of blocks
  - `kernel_size`: Kernel size used in the blocks
  - `expand_ratio`: Expansion ratio of intermediate channels (determined from a table based on `num_blocks`)
  - `num_out_channels`: Number of output channels in a block

The supported block architecture is `MBConv`, commonly used in models such as EfficientNet.

#### Search Step

In the search step, multi-objective optimization is performed using two metrics: hardware cost and model accuracy.

**Obtaining Hardware Cost**
- The backend uses `PPAEstimator (APM)` to measure hardware costs.
- The system connects to the `PPAEstimator Server` via the network using `gRPC` to send estimation requests and receive results.
- For object detection tasks, since the neural network head has a dynamic shape, only the structures measurable by APM are extracted and evaluated.

**Supported Metrics**
- **Hardware cost metrics:**
  - `latency`
  - `energy`
- **Accuracy metrics:**
  - For classification tasks: `accuracy/top1`
  - For object detection tasks: `coco/bbox_mAP`

**Search Algorithm**

- The backend uses [Optuna](https://optuna.org/) as the search framework.
- The search algorithm employed is `NSGA-III`, a type of evolutionary multi-objective optimization algorithm.

## Configuration Files

### Common Configuration Files

The training and search steps each require their own configuration files, both of which inherit settings from the following common configuration files:

**For Classification Tasks:**

- `configs/nas/mmcls/_base_/nas_backbones/compbignas_mobilenetv2_supernet.py`
  - Configuration file for the search space
- `configs/nas/mmcls/_base_/settings/imagenet_compbignas.py`
  - Configuration file for the dataset and parameter schedules

**For Object Detection Tasks:**

- `configs/nas/mmdet/_base_/nas_backbones/compbignas_mobilenetv2_supernet.py`
  - Configuration file for the search space
- `configs/nas/mmdet/_base_/settings/coco_compbignas.py`
  - Configuration file for the dataset and parameter schedules

**For Semantic Segmentation Tasks:**

- `configs/nas/mmseg/_base_/nas_backbones/compbignas_mobilenetv2_supernet.py`
  - Configuration file for the search space
- `configs/nas/mmdet/_base_/settings/cityscapes_compbignas.py`
  - Configuration file for the dataset and parameter schedules 

The next section provides a detailed explanations of these configuration files.

#### Search Space Configuration Files

When opening the configuration file, several variables are defined at the root level. Their descriptions are as follows:

- **`arch_setting`**: Configuration for the search space. Modify this section to customize the search space. (Details provided later)
- **`custom_groups`**: Settings for applying search parameters to the system. No modifications are required.
- **`input_resizer_cfg`**: Configuration for input resolution in the search space. Modify this to customize the search space. (Details provided later)
- **`nas_backbone`**: Configuration that applies the `arch_setting` defined above to the SuperNet. No modifications are required.

**arch_setting** and **input_resizer_cfg**

These settings correspond to the search parameters described in the previous section.

- `arch_setting.kernel_size`
  - Defines the search range for kernel sizes in MBConv blocks.
  - Each element in the list follows the format `[min_kernel_size, max_kernel_size, step]`, representing the range and step size for kernel sizes.
  - Example: `[3, 5, 2]` means that the search space includes kernel sizes `3` and `5`.

- `arch_setting.num_blocks`
  - Defines the search range for the number of repetitions (blocks) in MBConv blocks.
  - Each element follows the format `[min_num_blocks, max_num_blocks, step]`.
  - Example: `[1, 4, 1]` means the search space includes `1`, `2`, `3`, and `4` blocks.

- `arch_setting.expand_ratio`
  - Defines the search range for the expansion ratio of intermediate channels in MBConv blocks.
  - Specified as a dictionary mapping `{num_blocks: expand_ratio}`.
  - Example: `{1: 4, 2: 6}` means `expand_ratio=4` when `num_blocks=1` and `expand_ratio=6` when `num_blocks=2`.

- `arch_setting.num_out_channels`
  - Defines the search range for the number of output channels in MBConv blocks.
  - Each element follows the format `[min_channels, max_channels, step]`.
  - Example: `[64, 128, 16]` means the search space includes `64`, `80`, `96`, `112`, and `128` channels.

- `arch_setting.first_num_out_channels`
  - Defines the search range for the number of output channels of the first convolution in the backbone.
  - Follows the format `[min_channels, max_channels, step]`.
  - Example: `[16, 32, 8]` means the search space includes `16`, `24`, and `32` channels.

- `arch_setting.last_num_out_channels`
  - Defines the search range for the number of output channels of the backbone.
  - Follows the format `[min_channels, max_channels, step]`.
  - Example: `[1280, 2048, 64]` means the search space includes:
    `1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048` channels.

- `input_resizer_cfg`
  - Defines the search range for the model's input resolution.
  - Specified directly as a list of resolution pairs.
  - Example: `[[height0, width0], [height1, width1]]` means the search space includes `[height0, width0]` and `[height1, width1]` resolutions.

#### Dataset and Schedule

The system utilizes the following frameworks:
- [mmclassification (mmpretrain)](https://github.com/open-mmlab/mmpretrain/tree/v1.0.0rc0)
- [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc0)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

Please refer to their respective documentation for detailed information on datasets and schedules.

### Training Configuration Files

In most cases, modifications to the training configuration files are unnecessary. However, adjustments may be required in the following situations:

**When Changing the Search Range (Backbone Output Channels)**

Modify `supernet.head.in_channels` to match the maximum size of `last_mutable_channels` from the search space.

**When Changing the Number of Classes (Dataset Change)**

Modify `supernet.head.num_classes` and `supernet.head.loss.num_classes` to reflect the new number of classes.

### Search Configuration Files

The search configuration file requires adjustments before execution. Below are situations where adjustments are necessary and the appropriate modifications:

**When Executing a Search**

Set `train_cfg.estimator.servers` with the address and port of the PPAEstimator Server.

Example: To use one server on `localhost` with port `50051`:

```
servers=[
    {
        "host": "localhost",
        "ports": [50051]
    }
],
```

Example: When parallelizing time-consuming HW cost estimation using multiple servers on different nodes with multiple ports (total 8 instances):

```
servers=[
    {
        "host": "localhost",
        "ports": [50051, 50052, 50053, 50054]
    },
    {
        "host": "xxx.xxx.xxx.xxx",
        "ports": [50051, 50052]
    },
    {
        "host": "xxx.xxx.xxx.yyy",
        "ports": [50051, 50052]
    },
],
```

---

**When Changing the Number of Search Iterations**

Modify `train_cfg.max_epochs` to set the desired number of iterations.

**When Modifying PPAEstimator Hardware Settings**

Modify `train_cfg.estimator.hw_config`.
Refer to the PPAEstimator documentation for details on configuration items.

**When Changing Optimization Targets**

Modify `train_cfg.cost_key` to specify the hardware cost metric to optimize.
Supported metrics include:
- `latency`
- `energy`
