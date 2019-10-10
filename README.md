## [Object Captioning and Retrieval with Natural Language](https://arxiv.org/pdf/1803.06152.pdf)
By [Anh Nguyen](https://sites.google.com/site/quanganh0104/), [Quang D. Tran] (https://ai.aioz.io/), Thanh-Toan Do, Ian Reid, Darwin G. Caldwell, Nikos G. Tsagarakis

![object_captioning](https://raw.githubusercontent.com/nqanh/object_captioning/master/data/git_img/1348947380.jpg "object_captioning")


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
- Download pretrained weight from [this link](https://drive.google.com/drive/folders/1DsUUDEUtxL9IE2u728qpzclAkOz45C1v?usp=sharing), and put it under your `$PROJECT_PATH\trained_weight` folder
- Download the [Flickr5k](https://sites.google.com/site/objcaptioningretrieval/) dataset, and put it under your `$PROJECT_PATH\data\VOCdevkit2007` folder
- Change the project path in file `lib/model/config.py`: `__C.root_folder_path = '$PROJECT_PATH'` 
- Build the lib module: `cd $PROJECT_PATH/lib` then `make`
- Run the demo: `cd $PROJECT_PATH/tool` then `python demo_caption.py` to generate captions for your images
	
	
### Training

1. We train the network on [Flickr5k](https://sites.google.com/site/objcaptioningretrieval/) dataset
	- We need to format Flickr5k dataset as in Pascal-VOC dataset for training.
	- For your convinience, we did it for you. Just download this file ([Google Drive](https://drive.google.com/file/d/1FIAvc9AsSGYEYQmvJ1zH51FhXPos8vEc/view?usp=sharing) and extract it into your `$PROJECT_PATH\data\VOCdevkit2007` folder.

2. Train the network:
	- `python $PROJECT_PATH/tool/trainval_net.py`


If you find this source code useful in your research, please consider citing:

	@article{Nguyen_objcaption,
	  author    = {Anh Nguyen and
				   Duy Q. Tran and
				   Thanh{-}Toan Do and
				   Ian D. Reid and
				   Darwin G. Caldwell and
				   Nikos G.Tsagarakis},
	  title     = {Object Captioning and Retrieval with Natural Language},
	  journal   = {International Conference on Computer Vision Workshop},
	  year      = {2019},
	}


### License
MIT License

### Acknowledgement
This repo used a lot of source code from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) and [AffordanceNet](https://github.com/nqanh/affordance-net)


### Contact
If you have any questions or comments, please send an email to: `a.nguyen@ic.ac.uk`

