

## Semantic Segmentation Experiment on Synthia

This folder contains files for semantic segmentation experiment on Synthia dataset.

## Data Preprocessing

Download raw Synthia dataset from <a href="http://synthia-dataset.net/downloads/">here</a>. Extract the `.rar` file to get the directory containing the following folders:
```
SYNTHIA-SEQS-01-DAWN/
SYNTHIA-SEQS-01-FALL/
SYNTHIA-SEQS-01-NIGHT/
SYNTHIA-SEQS-01-SUMMER/
SYNTHIA-SEQS-01-WINTER/
SYNTHIA-SEQS-02-DAWN/
SYNTHIA-SEQS-02-FALL/
SYNTHIA-SEQS-02-NIGHT/
SYNTHIA-SEQS-02-RAINNIGHT/
SYNTHIA-SEQS-02-SOFTRAIN/
SYNTHIA-SEQS-02-SUMMER/
SYNTHIA-SEQS-02-WINTER/
SYNTHIA-SEQS-04-DAWN/
SYNTHIA-SEQS-04-FALL/
SYNTHIA-SEQS-04-NIGHT/
SYNTHIA-SEQS-04-RAINNIGHT/
SYNTHIA-SEQS-04-SOFTRAIN/
SYNTHIA-SEQS-04-SUMMER/
SYNTHIA-SEQS-04-WINTERNIGHT/
SYNTHIA-SEQS-04-WINTER/
SYNTHIA-SEQS-05-FOG/
SYNTHIA-SEQS-06-SPRING/
SYNTHIA-SEQS-06-SUNSET/
```
To preprocess the data, `cd` into `data_prep` directory. In the `command_gen_pc_with_label.sh`, make `data_root` to be the directoy containing all raw folders and `output_dir` to be the output directory. Then execute
```
sh command_gen_pc_with_label.sh
```
The generated processed data is also provided <a href="https://drive.google.com/file/d/1nGgHVofbmNbzaKYbRe9RxUWxGuyXe3HP/view?usp=sharing">here</a> for download (~5.1GB).

Then execute `python gen_label_weights.py --data_root ../processed_pc` to get the label weight file `data_prep/labelweights.npz`.

## Training of Direct Grouping

The script for training and testing of direct grouping model is `command_train_direct.sh`. To train, use the following command.

```
sh command_train_direct.sh
```

One may change the flags such as `num_frame`, `num_point` etc for different architecture specs.

