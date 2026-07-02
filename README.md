# GW-VRWKV
## Requirements

- **OS**: Linux (recommended for compatibility)
- **Python**: 3.8
- **PyTorch**: 1.11.1+cu118

> We have tested the code under the above environment. 

---

## Data Processing

1. Download the publicly available HSI datasets and place them in the corresponding directory refer to the dataload function in the ``utils\data_load_operate.py".
---

## Model
Our model is located in the `/model` folder. The main network structure is placed in "GW_SWKV. py".

---

## Training
1. During the training phase, it is necessary to run the train script and import the corresponding configuration file. Taking the Hanchuan dataset as an example, executing

"python -u train_GW-RWKV.py --config configs_619abla/sw_hanchuan_full.yaml" 

can start training, and specific parameters can refer to the corresponding YAML file under ``configs_619abla".

2. We also built a complete training script to train the given four HSI datasets in sequence. You can use the content in ``run_full_train_619abla.sh".

Note: Due to the use of custom CUDA functions to implement the VRWKV operator, direct training in step 1 may prompt CUDA related issues. You can refer to ``run_full_train_619abla.sh" and combine it with your environment for processing.
