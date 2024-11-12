# Shadow_Removal-
This repository contains code for removing shadows from images. I participated in the CVPR workshop challenge, where I achieved 7th rank worldwide and published a report.



## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
git clone https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~

running eval

python main.py --mode test --data_dir path/to/input/dir  --batch_size 1 --test_model ./pretrained/model.pkl --gpu_id 0

