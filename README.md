<div align="center">
<!-- <p align="center"> <img src="./assets/EMAGE_2024/1711449143651.jpg" width="100px"> </p> -->
<h2>TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio-Motion Embedding and Diffusion Interpolation</h2>

<a href='https://pantomatrix.github.io/TANGO/'><img src='https://img.shields.io/badge/Project-TANGO-blue' alt='Project'></a>
<a href='https://www.youtube.com/watch?v=_DfsA11puBc'><img src='https://img.shields.io/badge/YouTube-TANGO-rgb(255, 0, 0)' alt='Youtube'></a>
<a href='https://huggingface.co/spaces/H-Liu1997/TANGO'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://arxiv.org/abs/2410.04221'><img src='https://img.shields.io/badge/Paper-ArXiv-yellow' alt='Project'></a>

</div>

# News

Welcome contributors! Feel free to submit the pull requests!

- **[2024/10]** Welcome to try our [TANGO](<(https://huggingface.co/spaces/H-Liu1997/TANGO)!>) on Hugging face space !
- **[2024/10]** Code for creating gesture graph is available.
- **[2024/10]** Video data download [Google Drive](https://drive.google.com/drive/folders/1-A_Bb_L7UMLvckbHmZRltnb7jgaLzd6D?usp=sharing) (show-oliver and harward business)

<p align=center>
<img src ="./assets/hg.png" width="60%" >
</p>

# Results Videos

<p align="center">
  <img src="./assets/demo8.gif" width="32%" alt="demo0">
  <img src="./assets/demo1.gif" width="32%" alt="demo1">
  <img src="./assets/demo2.gif" width="32%" alt="demo2">
</p>
<p align="center">
  <img src="./assets/demo3.gif" width="32%" alt="demo3">
  <img src="./assets/demo5.gif" width="32%" alt="demo5">
  <img src="./assets/demo0.gif" width="32%" alt="demo6">
</p>
<p align="center">
  <img src="./assets/demo7.gif" width="32%" alt="demo7">
  <img src="./assets/demo6.gif" width="32%" alt="demo8">
  <img src="./assets/demo9.gif" width="32%" alt="demo9">
</p>

# Demo Video (on Youtube)

<p align=center>
    <a href="https://youtu.be/xuhD_-tMH1w?si=Tr6jHAhOR1fxWIjb">
    <img  width="68%" src="./assets/video.png">
    </a>
</p>

# 📝 Release Plans

- [x] Training codes for AuMoClip
- [x] Processed Youtube Buiness Video data (very small, around 15 mins)
- [x] Scripts for creating gesture graph
- [x] Inference codes with AuMoClip and pretrained weights

# ⚒️ Installation

## Clone the repository

```shell
git clone https://github.com/CyberAgentAILab/TANGO.git
cd TANGO
```

## Build Environment

For inference and training CLIP part, we recommend a python version `==3.10.16` and cuda version `==11.8`. Now HuggingFace Space version is py310 version:

```shell
# [Optional] Create a virtual env
conda create -n tango_py310 python==3.10.16
conda activate tango_py310
# Install with pip:
python -m pip install -r ./pre-requirements.txt
python -m pip install -r ./requirements.txt
```

# 🚀 Training and Inference

## Inference

Here is the command for running inference scripts under the path `<your root>/TANGO/`, it will take around 3 min to generate two 8s videos. You can visualize by directly check the video or check the result .npz files via blender using our blender addon in [EMAGE](https://github.com/PantoMatrix/PantoMatrix).

_Necessary checkpoints and pre-computed graphs will be automatically downloaded during the first run. Please ensure that at least 10GB of disk space is available._

```shell
# inference 
python inference.py --audio_path ./datasets/cached_audio/example_male_voice_9_seconds.wav --character_name ./datasets/cached_audio/speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4

# start gradio app like hugging face space
python app.py
```

## Training JointEmbedding (CLIP)

```shell
# download the training data from https://drive.google.com/file/d/11ZQI8mB7mP8OtlIdcjtxKvg7OxVZ4t7d/view?usp=drive_link

torchrun --nproc_per_node=1 train_high_env0.py --config ./configs/baseline_high_env0.yaml
```

### Create the graph for custom character

For building a motion graph, we recommend a python version `==3.9.20` and cuda version `==11.8` to support `mmcv` and `mmpose`. 

```shell
# [Optional] Create a virtual env
conda create -n tango_py39 python==3.9.20
conda activate tango_py39
# Install with pip:
python -m pip install -r ./pre-requirements_py39.txt
python -m pip install -r ./requirements_py39.txt
```

```shell
# set up the py39
python create_graph.py
```

# Copyright Information

We thank the open-source project [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [FiLM](https://github.com/caffeinism/FiLM-pytorch), [SMPLerX](https://github.com/caizhongang/SMPLer-X).

Check out our previous works for Co-Speech 3D motion Generation <a href="https://github.com/PantoMatrix/PantoMatrix">DisCo, BEAT, EMAGE</a>.

This project is only for research or education purposes, and not freely available for commercial use or redistribution. The script is available only under the terms of the [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode) (CC BY-NC 4.0) license.
