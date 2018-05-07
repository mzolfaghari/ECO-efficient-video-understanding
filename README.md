
#### Code and models of [paper](https://arxiv.org/pdf/1804.09066.pdf). " ECO: Efficient Convolutional Network for Online Video Understanding" 
 By Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox


### Update
- **2018.4.17**: Repository for ECO.


### Introduction
This repository will contains all the required models and scripts for the paper [ECO: Efficient Convolutional Network for Online Video Understanding](https://arxiv.org/pdf/1804.09066.pdf).

![](doc_files/s_model.png)


In this work, we introduce a network architecture that takes long-term content into account and enables fast per-video processing at the same time. The architecture is based on merging long-term content already in the network rather than in a post-hoc fusion. Together with a sampling strategy, which exploits that neighboring frames are largely redundant, this yields high-quality action classification and video captioning at up to 230 videos per second, where each video can consist of a few hundred frames. The approach achieves competitive performance across all datasets while being 10x to 80x faster than state-of-the-art methods.


### Results
Action Recognition on UCF101 and HMDB51           |  Video Captioning on MSVD dataset
:-------------------------:|:-------------------------:
![](doc_files/s_fig1.png)  |  ![](doc_files/s_fig2.png)



### TODO
1. Code and Models
2. Data
3. Tables and Results
4. Demo


### Contact

  [Mohammadreza Zolfaghari](https://github.com/mzolfaghari/ECO_efficient_video_understanding)

  Questions can also be left as issues in the repository. We will be happy to answer them.
