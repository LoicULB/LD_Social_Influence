# LD_Social_Influence

## Introduction
This repository was used to attempt to reproduce a scientific paper : "Social Influence as Intrinsic Motivation for Deep 
Reinforcement Learning" by : Natasha Jaques, Angeliki Lazaridou, Edward Hughes, Caglar Gulcehre, Pedro Ortega, 
Dj Strouse, Joel Z. Leibo andNando De Freitas 

The original paper presents a way to model social influence in a deep reinforcement learning algorithm. 
The results shows the effect of giving rewards to agents who have a influence on the other agents action.
The social influence used in conjuction with an A3C learning algorithm

However, due to multiple difficulties (the code of the paper was difficult to setup, our firt implementations did not 
achieve sufficient results), we did not succeed into making a sufficiently good A2C implementation from scratch. 
This repo thus contains our first two attempts at writing an A2C algorithm, a way to see how to use the RLlib library 
used in the paper and finally a code that we used to compare the performances of A2C vs A3C by using the some codes of 
the paper
## Setup 
To be able to run our multiple programs : follow the instructions
> git clone https://github.com/LoicULB/LD_Social_Influence.git \
> cd LD_Social_Influence

Now, either create a new virtual environment via Pycharm or other IDEA that have the same functionality or you can do it 
manually like the following :
> python3 -m venv LD_A2C_Env \
> pip install -r requirements.txt

Now you can launch all our implementions that do not have dependency with the original paper's code.
Read the Structure section to see which python file to execute depending on what you want to do.
## Original paper's code
We had a lot of difficulties to make the environment work on some of our computers. Even then we did not manage to make
it work on all of our machines. We found no way to do it on Windows, we succeeded on a Mac after many trials.
The ubuntu was the OS where it was the easier to make the original paper"s code run. 
So do not be suprised if the setting up do not work on your computer.
### Set up the original code
#### Setup from the original rep
I you are lucky, maybe the instalation guide from the original code guide will work.
Here it is (we advise you to do it on another folder than the one of our github) :
> git clone -b master https://github.com/eugenevinitsky/sequential_social_dilemma_games \
> cd sequential_social_dilemma_games \
> python3 -m venv venv # Create a Python virtual environment \
> . venv/bin/activate \
> pip3 install --upgrade pip setuptools wheel \
> python3 setup.py develop \
> pip3 install -r requirements.txt \
> . ray_uint8_patch.sh 
> cd run_scripts 

The original code github : https://github.com/eugenevinitsky/sequential_social_dilemma_games
#### Rohan Pull Request
If the preceeding installations did not worked, then try the one proposed by another pull request : 
https://github.com/Rohan138/sequential_social_dilemma_games

For it you will need to install Anaconda : https://www.anaconda.com/products/individual

The setup instructions given in his repository are misleading so just try to follow those :
> conda create -n ssd2 python==3.8.10
conda activate ssd2 \
>  . conda_uint8_patch.sh  \
> pip3 install -r requirements.txt \
python3 setup.py develop \
. venv_uint8_patch.sh \
cd run_scripts/ \
python3 train.py

If after launching python3 train.py, you have import errors , with "requests" for exemple, pip install those modules and 
it should work at the end. If the error given is from "a3c.optimizer", then try to launch train.py via an IDEA such as 
Pycharms.
Once you see that train.py seems to run, you can kill the process and follows the instruction of the custom_train.py 
section.
If it still do not work, you can contact the following mail address : Loic.Quivron@ulb.be and we will try to help you
if possible
### custom_train.py
Now that the original code environment is setted up, you can put the "custom_train.py" file in the
./run_scripts/ directory and try to run it via :
> python3 custom_train.py

This file is used to launch training with A2C and A3C implementation from RLLib and compare their results. Modify the 
hyperparameters if you want to reproduce our plots :) 
## Structure

### Social_dilemma
This folder contains every files used to implement the Harvest Environment (one of the environments studied in the 
original paper). It was directly taken from the paper's code.

### A2C_Phil
A directory where we used a base A2C implementation from a youtube tutorial by "Machine Learning with Phil" using 
Keras (https://www.youtube.com/watch?v=LawaN3BdI00 ). The code was a bit adapted to work on the Harvest Environment
(Multi Agent Environment), since the base code was designed for CartPole (a Single Agent environment). 
No sufficient results were obtained and we had insight that the algorithm implemented could have be inadapted to our 
Harvest Environment so looked at another tutorial using Pytorch this time. 
The Neural Network is used on the run folder. The latter will be placed in A2C_Phil directory in next commits.

### PytorchCartPole
This directory was used to contain the implementation of the A2C learning algorithm from the following tutorial :
https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

It did perform well on CartPole and by executing the main.py file, you will see the training of an A2C agent on the
CartPole environment (this is the file used to compute figure XXX from our paper)

> python3 PytorchCartPole/main.py

### PytorchHarvest 
This directory contains our adaptation of the previous code to the Harves Environment. We have a single agent version and
a multi agent one.

TODO
### Run
The files that we will use to train our agents
TODO

### rayTuto
Used to have a simple example on how to use RLlib A2C implementation
TODO

### custom_training.py file
TODO
