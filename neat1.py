import gym
import numpy as np
import neat
from neat.parallel import ParallelEvaluator
import cv2
import visualize
from gym.wrappers import RecordVideo
from functools import partial
import random

# Initialize the gym environment for MsPacman
episode_trigger = lambda x: x % 100 == 0  # Record video every 10 episodes
env = RecordVideo(gym.make('ALE/MsPacman-v5', render_mode='rgb_array'), 'video', episode_trigger=episode_trigger)


def _eval_genomes(evaluate_agent, genomes, config):
    parallel_evaluator = ParallelEvaluator(8, eval_function=evaluate_agent)
    parallel_evaluator.evaluate(genomes, config)


def eval_single_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0.0
    for i in range(1):
        observation, _= env.reset()
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = observation.flatten()
        temp = net.activate(observation)
        action  = np.argmax(temp)
        if action == 0:
            action = random.randint(1,8)
            #temp.pop(0)
            #action = np.argmax(temp) + 1
        done = False
        while not done:
            observation, reward, done, truncated, info = env.step(action)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = observation.flatten()
            temp = net.activate(observation)
            action = np.argmax(temp)
            if action == 0:
                action = random.randint(1,8)
                #temp.pop(0)
                #action = np.argmax(temp) + 1
            fitness += reward
            if done:
                break
    return fitness


# Define the fitness function for evaluating the agents
def evaluate_agent(genomes, config):
    for genome_id, genome in genomes:
        observation, _ = env.reset()
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = observation.flatten()

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        done = False
        while not done:
            action = np.argmax(net.activate(observation))
            if action == 0:
                action = random.randint(1,8)
            observation, reward, done, truncated, info = env.step(action)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = observation.flatten()

            fitness += reward
        genome.fitness = fitness
        visualize.draw_net(config, genome, filename='net')


# Define the NEAT configuration parameters
config_path = 'config.txt'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Initialize the NEAT population
population = neat.Population(config)

# Add reporters to track the progress of the NEAT algorithm
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

# Train the agents using the NEAT algorithm
stats = neat.StatisticsReporter()
# Parallel training
best_genome = population.run(partial(_eval_genomes, eval_single_genome), 200)
#visualize.draw_net(config, best_genome, filename='net')
# Not parallel training
# best_genome = population.run(evaluate_agent, 100)
#visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")
#visualize.plot_species(stats, view=False, filename="species.svg")

# Evaluate the best agent
best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
observation, _ = env.reset()
observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
observation = observation.flatten()

done = False
while not done:
    action = np.argmax(best_net.activate(observation))
    observation, reward, done, truncated, info = env.step(action)
    # env.render()

# Close the environment
env.close()

