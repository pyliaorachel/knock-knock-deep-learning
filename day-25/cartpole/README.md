# OpenAI Gym CartPole Example
 
Demonstration of the following algorithms to solve the cart pole problem in OpenAI gym:

- Random action
- Hand-made policy
- Q-table
- Deep Q-network

Modified from [OpenAI gym docs](https://gym.openai.com/docs/), [Q-table example](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947), [莫煩 Python DQN](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py).

## Usage

```bash
# Create & activate conda environment
$ conda env create -f env.yml
$ conda activate cartpole-tutorial

# Run
$ python <algorithm>/main.py # algorithm: random_action, hand_made_policy, q_table, dqn
```
