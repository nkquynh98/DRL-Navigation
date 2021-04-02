[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, we have to train an agent to navigate in a stochastic environment and collect as many as yellow bananas as possible while avoiding the purple banana. The basic Deep Q-learning developed by Mnih et. al. [1] is implemented in this project to solve that problem.

The code and the framework of this project is based on the code of the DQN implemetation from Udacity [2]. 

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the this GitHub repository, in the `/DRL-Navigation` folder, and unzip (or decompress) the file. 

3. Follow the steps in the original [DRLND repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)  to download the required dependencies and set up an Anaconda environment with Python = 3.6. CAUTION: The python version must be `3.6` to avoid any confliction with the Unity Agent version of `0.4.0`. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent. You can also run a pre-trained agent to evaluate its performance.

Some important files:
* `Navigation.ipynb` --> The training and testing process.
* `agent.py` --> The DQN agent that handles the learning process.
* `DQN_model.py` --> The Deep Q-learning Network architecture.
* `Checkpoint_DQN` --> The pre-trained parameters of the DQN.
* `REPORT.md` --> The report for this project.
## Reference
* [1] Mnih, Volodymyr & Kavukcuoglu, Koray & Silver, David, et. al.. [*Human-level control through deep-reinforcement learning*](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [2] [DQN lunar-lander implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution)
