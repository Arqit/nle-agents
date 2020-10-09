import gym
import nle
env = gym.make("NetHackScore-v0")
env.reset()  # each reset generates a new dungeon
env.step(1)  # move agent '@' north
env.render()
print(env.reset())
