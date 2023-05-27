# Self-Attention-Meta-Learner-for-Continual-Learning
This is the official PyTorch implementation for the [Self-Attention Meta-Learner for Continual Learning (Sokar et al., AAMAS 2021)](https://arxiv.org/abs/2101.12136) paper at the 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2021).

We propose a new method, named Self-Attention Meta-Learner (SAM), which learns a prior knowledge for continual learning that permits learning a sequence of tasks, while avoiding catastrophic forgetting. SAM incorporates an attention mechanism that learns to select the particular relevant representation for each future task. Each task builds a specific representation branch on top of the selected knowledge, avoiding the interference between tasks.


# Requirements
* Python 3.6
* Pytorch 1.2
* torchvision 0.4

# Usage
You can use main.py to run SAM on the split MNIST benchmark. 

```
python main.py
```

# Reference
If you use this code, please cite our paper:
```
@inproceedings{sokar2021selfattention,
  title={Self-Attention Meta-Learner for Continual Learning},
  author={Ghada Sokar and Decebal Constantin Mocanu and Mykola Pechenizkiy},
  booktitle={20th International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2021},
  year={2021},
  organization={International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS)}
}
```

# Acknowledgments
We adapt the source code from the following repository to train the meta-learner model (prior knowledge)

[MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch)


