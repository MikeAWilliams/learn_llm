"""
Follow allong with the cart pole learning example found
https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/22-DeepRL/CartPole-RL-PyTorch.ipynb?utm_source=chatgpt.com
"""

import torch
from torch import nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

# render_model="human" causes every env.step to render the environment
env_vis = gym.make("CartPole-v1", render_mode="human")
env_train = gym.make("CartPole-v1")

print(f"Action space: {env_vis.action_space}")
print(f"Observation space: {env_vis.observation_space}")


# number of inputs matches the obs above: postion_x, velocity_x, angle, angular_velocity
num_inputs = 4
# number of outputs matches the action space above: move_left, move_right
num_actions = 2

model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_actions, bias=False, dtype=torch.float32),
    # dim=1 because batch size is 1
    torch.nn.Softmax(dim=1),
)


def run_episode(max_steps_per_episode=10000, output=False, render=False):
    if render:
        env = env_vis
    else:
        env = env_train
    states, actions, probs, rewards = [], [], [], []
    state, info = env.reset()
    for _ in range(max_steps_per_episode):
        # state is a numpy array shape (4,)
        # np.expand_dims(state,0) mean create a tensor of shape (1,4)
        # axis=0 means create a new dimension at the beginning
        # pytorch expexts a tensor of shape (batch_size, num_inputs)
        # note indexing into the tensor at 0 returns a 1d tensor, in this case with 2 probabilities
        action_probs = model(torch.from_numpy(np.expand_dims(state, 0)))[0]
        # note detach removes the tesnor from pythorch compute graph, this prevents gradient tracking
        # sampling is not differentiable anyway
        # .numpy() converts the tensor to a numpy array
        # squeeze strips any dimensions of size 1
        # finally choise picks randomly based on the probabilities
        action = np.random.choice(
            num_actions, p=np.squeeze(action_probs.detach().numpy())
        )
        nstate, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break
        states.append(state)
        actions.append(action)
        probs.append(action_probs.detach().numpy())
        rewards.append(reward)
        state = nstate
    if output:
        print(
            f"terminated by after {len(states)} steps, terminated {terminated}, truncated {truncated}"
        )
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


# Example
# rewards = [1, 1, 1]
# gamma = 0.99
# before normalization
# ret = [
#   1 + 0.99 + 0.99² = 2.9701,
#   1 + 0.99 = 1.99,
#   1
# ]
# Mean ≈ 1.99
# Std ≈ 0.81
# So after normalization
# [(2.97-1.99)/0.81,
#  (1.99-1.99)/0.81,
#  (1-1.99)/0.81]
# ≈ [1.2, 0, -1.2]
# This weights earlier actions higher than later and also
# This has the efect of changing the reward to say "Was this action better or worse than the average action in the epoch"
eps = 0.0001


def discounted_rewards(rewards, gamma=0.99, normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret - np.mean(ret)) / (np.std(ret) + eps)
    return ret


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train_on_batch(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions) * y)
    loss.backward()
    optimizer.step()
    return loss


print("simulation before we train")
# run_episode(render=True)
run_episode(output=True)  # render takes a long time

alpha = 1e-4

history = []
for epoch in range(300):
    states, actions, probs, rewards = run_episode()
    # recall that an action is 0 or 1 based on the index of the model output selected by probability sample
    # we had an array of actions but we ran np.vstack(action) which makde it into a list of lists where each internal list had one element
    # T transposes the array making it into a list with one list inside and all the elements in that
    # taking out the 0 element grabs that internal list
    # np.eye(2) converts a numeric value with 2 possible values into a 2x2 matrix one hot encoded
    # value of 0 to [1, 0] and value of 1 to [0, 1]
    one_hot_actions = np.eye(2)[actions.T][0]
    # a probability row is a list of two probabilities, after 1 hot encoding above we end up
    # with the action taken as a 1. so a single subtraction is like [1, 0] - [0.7, 0.3] = [0.3, -0.3]
    gradients = one_hot_actions - probs
    dr = discounted_rewards(rewards)
    # weight the gradient by dicsounted rewards
    gradients *= dr
    # target here is not a labeled correct value, just a nudge to the model
    # because alpha is small it will take some big rewards to make target much different than the initial probs
    # so when the rewards are small we don't change thigns much
    target = alpha * np.vstack([gradients]) + probs
    train_on_batch(states, target)
    history.append(np.sum(rewards))
    if epoch % 100 == 0:
        print(f"{epoch} -> {np.sum(rewards)}")


print("simulation after we train")
# run_episode(render=True)
run_episode(output=True)  # render takes a long time
