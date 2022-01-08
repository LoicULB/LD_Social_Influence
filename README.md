# LD_Social_Influence

## Introduction
This repository was used to attempt to reproduce a scientific paper : "Social Influence as Intrinsic Motivation for Deep 
Reinforcement Learning" by : Natasha Jaques, Angeliki Lazaridou, Edward Hughes, Caglar Gulcehre, Pedro Ortega, 
Dj Strouse, Joel Z. Leibo andNando De Freitas 

The paper presents a way to model social influence in a deep reinforcement learning algorithm. 
The results shows the input of giving rewards to agents who have a influence on the other agents action.
The social influence used in conjuction with an A3C learning algorithm

However, due to multiple difficulties (the code of the paper was difficult to setup, our firts implementations did not 
achieved sufficient results), we did not succeed into making a sufficiently good A2C implementation from scratch. 
This repo thus contained our first two attempts at write an A2C algorithm, a way to see how to use the RLlib library 
used in the paper and finally a code that we used to compare the performances of A2C vs A3C by using the some codes of 
the paper

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
## Original paper's code
TODO
### Set up the original paper's code
TODO
### custom_train.py
TODO