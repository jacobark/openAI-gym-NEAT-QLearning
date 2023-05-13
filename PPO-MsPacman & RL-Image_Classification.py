#!/usr/bin/env python
# coding: utf-8

# In[88]:


import imageio
import numpy as np

from stable_baselines3 import PPO

model = PPO("MlpPolicy", "ALE/MsPacman-v5").learn(2_000_000)

images = []
obs = model.env.reset()
img = model.env.render(mode="rgb_array")
done = False
while done != True:
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, done,_ = model.env.step(action)
    img = model.env.render(mode="rgb_array")

imageio.mimsave("MsPacman_DQN.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], duration=35)


# # PPO for Image Classification on MNIST

# In[48]:


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
import time
import gym
import random
#from stable_baselines3 import bench


# In[49]:


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)


# In[50]:


class MnistEnv(gym.Env):
    def __init__(self, images_per_episode=1, dataset=(x_train, y_train), random=True):
        super().__init__()

        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(28, 28, 1),
                                                dtype=np.float32)

        self.images_per_episode = images_per_episode
        self.step_count = 0

        self.x, self.y = dataset
        self.random = random
        self.dataset_idx = 0

    def step(self, action):
        done = False
        reward = int(action == self.expected_action)

        obs = self._next_obs()

        self.step_count += 1
        if self.step_count >= self.images_per_episode:
            done = True

        return obs, reward, done, {}

    def reset(self):
        self.step_count = 0

        obs = self._next_obs()
        return obs

    def _next_obs(self):
        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y[next_obs_idx])
            obs = self.x[next_obs_idx]

        else:
            obs = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])

            self.dataset_idx += 1
            if self.dataset_idx >= len(self.x):
                raise StopIteration()

        return obs


# In[51]:


def mnist_ppo():
    logger.configure('./logs/mnist_ppo', ['stdout', 'tensorboard'])
    env = DummyVecEnv([lambda: Monitor(MnistEnv(images_per_episode=1), './logs/mnist_ppo')])

    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=32,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0001,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        tensorboard_log="./logs/mnist_ppo",
        verbose=1,
    )
    
    model.learn(total_timesteps=int(1.2e5))
    return model

start_time = time.time()
ppo_model = mnist_ppo()
print("PPO Training Time:", time.time() - start_time)


# In[56]:


def mnist_ppo_eval(ppo_model):
    attempts, correct = 0,0

    env = DummyVecEnv([lambda: MnistEnv(images_per_episode=1, dataset=(x_test, y_test), random=False)])
    
    try:
        while True:
            obs, done = env.reset(), [False]
            while not done[0]:
                obs, rew, done, _ = env.step(ppo_model.predict(obs)[0])

                attempts += 1
                if rew[0] > 0:
                    correct += 1
    except StopIteration:
        print()
        print('validation done...')
        print('Accuracy: {0}%'.format((float(correct) / attempts) * 100))

mnist_ppo_eval(ppo_model)


# # Running Q-Learning on MNIST

# In[84]:


def mnist_QL():
    logger.configure('./logs/mnist_ql', ['stdout', 'tensorboard'])
    env = DummyVecEnv([lambda: Monitor(MnistEnv(images_per_episode=1), './logs/mnist_ql')])
    Q = np.zeros([28*28, 10])
    alpha = .1
    G = 0
    for episode in range(1,1000000):
        done = False
        G, reward = 0,0
        state = env.reset()
        while done != True:
                if random.random() < (10 / (episode*.1)):  # take less random steps as you learn more about the game
                    action = random.randint(0,env.action_space.n-1)
                else:
                    action = np.argmax(Q[state])
                state2, reward, done, info = env.step([action])  # 2
                Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
                G += reward
                state = state2
    return Q

start_time = time.time()
ql_model = mnist_QL()
print("Q-Learning Training Time:", time.time() - start_time)


# In[87]:


def mnist_QL_eval(ql_model):
    attempts, correct = 0,0

    env = DummyVecEnv([lambda: MnistEnv(images_per_episode=1, dataset=(x_test, y_test), random=False)])
    
    try:
        while True:
            obs, done = env.reset(), [False]
            while not done[0]:
                action = np.argmax(ql_model[obs[:,:,0].ravel().astype(int)])
                obs, rew, done, _ = env.step([action])  # wrap action in an array

                attempts += 1
                if rew[0] > 0:
                    correct += 1
    except StopIteration:
        print()
        print('validation done...')
        print('Accuracy: {0}%'.format((float(correct) / attempts) * 100))

mnist_QL_eval(ql_model)


# In[ ]:




