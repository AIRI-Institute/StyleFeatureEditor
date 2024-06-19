#!/bin/bash

# conda create -n sfe python=3.10


conda install ninja -y
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt