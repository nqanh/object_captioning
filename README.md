## [Object Captioning and Retrieval with Natural Language](https://arxiv.org/pdf/1803.06152.pdf)
By Anh Nguyen, Thanh-Toan Do, Ian Reid, Darwin G. Caldwell, Nikos G. Tsagarakis


### Contents
1. [Requirements](#requirements)
2. [Quick Demo](#demo)
3. [Training](#training)

### Requirements

1. Tensorflow (version > 1.0)
2. Hardware
	- A gpu with ~6GB


### Quick Demo
- Clone the repo to your `$PROJECT_PATH` folder
- Download pretrained weight from [this link](#), and put it under your `$PROJECT_PATH\trained_weight` folder
- Run `python $Root\evaluation\demo_objcaption_net.py` to generate captions for your images
	
	
### Training

1. We train the network on [Flickr5k](https://sites.google.com/site/objcaptioningretrieval/) dataset
	- We need to format Flickr5k dataset as in Pascal-VOC dataset for training.
	- For your convinience, we did it for you. Just download this file ([Google Drive](https://drive.google.com/file/d/1FIAvc9AsSGYEYQmvJ1zH51FhXPos8vEc/view?usp=sharing) and extract it into your `$PROJECT_PATH\data` folder.

2. Train the network:
	- `python PROJECT_PATH/tool/trainval_net`


If you find this source code useful in your research, please consider citing:

	@article{DBLP:journals/corr/abs-1803-06152,
	  author    = {Anh Nguyen and
				   Thanh{-}Toan Do and
				   Ian D. Reid and
				   Darwin G. Caldwell and
				   Nikos G.Tsagarakis},
	  title     = {Object Captioning and Retrieval with Natural Language},
	  journal   = {CoRR},
	  volume    = {abs/1803.06152},
	  year      = {2018},
	  url       = {http://arxiv.org/abs/1803.06152},
	}


### License
MIT License

### Acknowledgement
This repo used a lot of source code from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) and [AffordanceNet](https://github.com/nqanh/affordance-net)


### Contact
If you have any questions or comments, please send an email to: `anh.nguyen@iit.it`

