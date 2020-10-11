

## Scene Flow Experiment on KITTI

This folder contains files for data and data processing scripts used in scene flow estimation experiments on KITTI dataset. The code for training and evaluating on the data remains to be released.

### Data Download

Download KITTI <a href="http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php">scene flow data</a> and <a href="http://www.cvlibs.net/datasets/kitti/raw_data.php">raw data</a>. Extract the `.zip` files such that the scene flow data directory and raw data directory looks like this

```
/path/to/sceneflow/
    devkit/
    testing/
    training/

/path/to/raw/
    2011_09_26/
    2011_09_29/
    2011_10_03/
    2011_09_28/
    2011_09_30/
```

### Data Processing

The script for processing of data is `gen_kitti_flow.py`. We extract four consecutive frames and generate the scene flow between the from the first to the second frame. It can be run by the following command

```
python gen_kitti_flow.py --kitti_sceneflow_dir /path/to/sceneflow --kitti_raw_dir /path/to/raw --output_dir output_folder
```

The processed data is also provided <a href="https://drive.google.com/file/d/1ui67lnfS0_clTehQwzDFwgMwUTsPG4jN/view?usp=sharing">here</a> for download (~270MB). 

