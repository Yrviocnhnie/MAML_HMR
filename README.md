This is a implementation of MAML algorithm using MAML as the base-model.

#### 1. Requirement:
Install torch and torchmeta to aviod conflicting 
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchmeta
```   
#### 2. Download Dataset 
Donwload Human3.6M following
- https://github.com/microsoft/MeshTransformer.git 

And run the data preprocess:
```
cd src_MAML/
python preprocess_human36m.py
```


### Training code/data
```
cd src_MAML/
python Meta_trainer_1.py
```


### Webcam Demo (thanks @JulesDoe!)
1. Download pre-trained models like above.
2. Run the demo
```
python demo.py
python plot.py
```



