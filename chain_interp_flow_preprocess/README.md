

## Pre-processing for Chained Flow Grouping Experiments

This folder contains files for pre-processing data for chained-flow grouping experiments. Files include scripts for off-line estimation of 3D scene flow, and scripts for interpolation and chaining of 3D scene flow for further frames.

## Download Point Cloud Data

### Synthia

For Synthia experiments, process the raw Synthia dataset or download the processed point clouds following instructions in <a href="https://github.com/xingyul/meteornet/blob/master/semantic_seg_synthia/README.md">here</a> (total size ~5.1GB). Then extract the files and place it like
```
../semantic_seg_synthia/processed_pc/
    SYNTHIA-SEQS-01-DAWN-000000.npz
    SYNTHIA-SEQS-01-DAWN-000001.npz
    ...
    SYNTHIA-SEQS-06-SUNSET-000841.npz
```

## Off-line Estimation of 3D Scene Flow for Consecutive Frames

In order to off-line estimate 3D scene flow for consecutive frames,
download the pre-trained FlowNet3D model from <a href="http://synthia-dataset.net/downloads/">this repo</a>, i.e. download model file from <a href="https://github.com/xingyul/meteornet/blob/master/semantic_seg_synthia/models/model_part_seg_meteor_direct.py">here</a> and checkpoints from <a href="https://drive.google.com/open?id=1Ko25szFFKHOq-SPryKbi9ljpOkoe69aO">here</a>. Make sure `log_train/` and `model_concat_upsa.py` are in this directory.

### Synthia

For Synthia experiments, run `command_pred_pairwise_flow_synthia.sh` by

```
sh command_pred_pairwise_flow_synthia.sh
```

The output scene flow will be stored in `../semantic_seg_synthia/init_flow`.

## Interpolate and Chain Scene Flow for Further Frames

The default number of nearest neighbors is set to be 2. One may change it to try with different settings.

### Synthia

For Synthia experiments, run `command_process_flow_to_chain_synthia.sh` by
```
sh command_process_flow_to_chain_synthia.sh
```

The output scene flow will be stored in `../semantic_seg_synthia/chained_flow`.






