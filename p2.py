import gym
import random
import numpy as np
from gym.wrappers import RecordVideo
import imageio

episode_trigger = lambda x: x % 2 == 0  # Record video every 1 episodes
env = RecordVideo(gym.make("ALE/MsPacman-v5", render_mode='rgb_array'), 'video', episode_trigger=episode_trigger)

Q = np.zeros([210*160, env.action_space.n])
alpha = .1
G = 0

for episode in range(1,500):
    done = False
    G, reward = 0,0
    state, temp = env.reset()
    while done != True:
            if random.random() < (10 / (episode*.1)):  # take less random steps as you learn more about the game
                action = random.randint(0,env.action_space.n-1)
            else:
                action = np.argmax(Q[state])
            state2, reward, done, truncated, info = env.step(action)  # 2
            Q[state, action] += alpha * (reward + np.max(Q[state2,:]) - Q[state, action])  # 3
            G += reward
            state = state2
    
    print('Episode {} Total Reward: {}'.format(episode, G))
images = []
state, temp = env.reset()
img = state
done = False
while not done:
    images.append(img)
    action = np.argmax(Q[state])
    obs, _, done, _, _ = env.step(action)
    img = obs

imageio.mimsave("MsPacman_Q_Learning.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], duration=35)
env.close()


