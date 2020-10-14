## MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences

Created by <a href="http://xingyul.github.io">Xingyu Liu</a>, <a href="https://scholar.google.com/citations?user=-S_9ZRcAAAAJ">Mengyuan Yan</a> and <a href="http://stanford.edu/~bohg">Jeannette Bohg</a> from Stanford University.

[[arXiv]](https://arxiv.org/abs/1910.09165) [[project]](https://sites.google.com/view/meteornet)

<img src="https://github.com/xingyul/meteornet/blob/master/doc/meteornet-teaser.png" width="60%">

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{liu2019meteornet, 
title={MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences}, 
author={Xingyu Liu and Mengyuan Yan and Jeannette Bohg}, 
booktitle={ICCV}, 
year={2019} 
}
```

## Abstract

Understanding dynamic 3D environment is crucial for robotic agents and many other applications. We propose a novel neural network architecture called MeteorNet for learning representations for dynamic 3D point cloud sequences. Different from previous work that adopts a grid-based representation and applies 3D or 4D convolutions, our network directly processes point clouds. We propose two ways to construct spatiotemporal neighborhoods for each point in the point cloud sequence. Information from these neighborhoods is aggregated to learn features per point. We benchmark our network on a variety of 3D recognition tasks including action recognition, semantic segmentation and scene flow estimation. MeteorNet shows stronger performance than previous grid-based methods while achieving state-of-the-art performance on Synthia. MeteorNet also outperforms previous baseline methods that are able to process at most two consecutive point clouds. To the best of our knowledge, this is the first work on deep learning for dynamic raw point cloud sequences.

## Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.9.0 GPU version, g++ 5.4.0, CUDA 9.0 and Python 3.5 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`. It's highly recommended that you have access to GPUs.

### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them first by `make` under each ops subfolder (check `Makefile`) or directly use `sh command_make.sh`. **Update** `arch` **in the Makefiles for different** <a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported">CUDA Compute Capability</a> **that suits your GPU if necessary**.

## Action Recognition Experiments on MSRAction3D

The code for action recognition experiments on MSRAction3D dataset is in `action_cls/`. Please refer to `action_cls/README.md` for more information on data preprocessing and experiments.

## Semantic Segmentation Experiments on Synthia

The code for semantic segmentation experiments on Synthia dataset is in `semantic_seg/`. Please refer to `semantic_seg/README.md` for more information on data preprocessing and experiments.

Note that only direct grouping models are released for now. Chain-flowed models will be released soon.

## Semantic Segmentation Experiments on KITTI

To be released. Stay tuned!

## Scene Flow Experiments

The code for data processing used in scene flow estimation experiments on KITTI dataset is in `scene_flow_kitti/`. Please refer to `scene_flow_kitti/README.md` for more information.

Stay tuned for other data and code for this part!

## License
Our code is released under MIT License (see LICENSE file for details).


## Related Projects

* <a href="https://arxiv.org/abs/1905.07853" target="_blank">Learning Video Representations from Correspondence Proposals
</a> by Liu et al. (CVPR 2019 Oral Presentation). Code and data released in <a href="https://github.com/xingyul/cpnet">GitHub</a>.
* <a href="https://arxiv.org/abs/1806.01411" target="_blank">FlowNet3D: Learning Scene Flow in 3D Point Clouds
</a> by Liu et al. (CVPR 2019). Code and data released in <a href="https://github.com/xingyul/flownet3d">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data released in <a href="https://github.com/charlesq34/pointnet2">GitHub</a>.



