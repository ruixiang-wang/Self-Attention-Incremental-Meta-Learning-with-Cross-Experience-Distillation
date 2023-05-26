# lifelong-object-recognition-challenge
This is a PyTorch implementation of baseline model of IROS2019 lifelong object recognition challenge ([link](https://lifelong-robotic-vision.github.io/competition/Object-Recognition.html)).
This repository provides:
* MobileNetV2 [[1](##References)]  backbone for the task:
  * input size: Using random crop for input images
* Pretrained models for specific batches: 
  * Training schemes: Two training methods (naive and cumulative) using backbone model are provided. Detailed definition of these two schemes are included in this  paper [[2](##References)].
  * Hyperparameters: We set 50 epochs for training and 30 epochs for finetuning each task with `learning_rate = 0.01`, `momentum=0.9`, `weight_decay=5e-4`,`batch_size=16`. 
  * Model files: Note that models after batch 3 and batch 5 for multitask (cumulative) training scheme (named `model_1.pth` and `model_2.pth`), model after batch 9 for finetuning (naive) scheme (named `model_3.pth`) are provided under `/model`. 
  * To run model: Feel free to load specific model in `evaluate.py`.
  * Please note that multitask (cumulative) training scheme is prohibited in the competition.
* Script for running the baseline model (with dataloader module for reference):
  * You can load the pretrained model and will generate results samples in your specified directory. 
  * The output files (in `csv`) are in the same format and name with the files to be submitted on our official online judgement platform CodaLab ([link](https://codalab.lri.fr/competitions/581)). The script will generate nine results csv files and our platform will compute the average precision score automatically. 


## Requirements
- The current version of the code has been tested with following libs:

  * `python 3.7`
  * `numpy 1.16.2`
  * `pandas 0.24.2`
  * `pytorch 1.1.0`
  * `torchvision 0.2.2`
  * `PIL 6.0.0`

- recommend install in virtual environment
```
$ conda create -n yourenvname python=3.7 anaconda
```
- Install the required the packages inside the virtual environment
```
$ source activate yourenvname
$ pip install -r requirements.txt
```


## Running the baseline model

Script for running the baseline model can be run with `evaluate.py`. Here `testset_path` and `output_path`is the path you stored your testset data and output prediction results for each batch. Results files will generated under specified output path. 
```
python3 evaluate.py testset_path output_path
```

## References

[1] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018. [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)]

[2] Lomonaco, Vincenzo, and Davide Maltoni. "CORe50: a New Dataset and Benchmark for Continuous Object Recognition." Conference on Robot Learning. 2017. [[pdf](https://arxiv.org/pdf/1705.03550)]




