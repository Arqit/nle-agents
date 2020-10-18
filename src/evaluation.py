import numpy as np
import gym
import nle
import random
from MyAgent import MyAgent, DQN
import torch

# Eventually, look into clipping the rewards (The justifacation being stated in the DeepMind DQN paper)
# There appraoch is working with rollouts... Rather focus on the framing aspect as opposed to the solution itself
# use their Resetting environment to be trained indefinitely... Otherwise, it will terminate after every episode
# It doesnt look like format_observations is doing the actual formatting (seems kinda useless)...
    # seems to just take the observation, transform to tensor and reshape...
    #It then returns the formatted obs. as a dictionary again
# update function is not implemented... Whatever calls it is a dictionary and it just adds the parameters to the the dictionary
# In line 350, its seems that _format_observations just creates the space and sorts of the obs. keys (glyphs, blstats)...
# Thereafter, we take this lamely formatted observation (dict) and adds the additional stuff to the observation from line 351
# The abovementioned is definitely the approach taken to build on the default state variables

#hey keep track of the provided state variables plus the following:
"""
result.update(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )
"""# Look into where is this used
# They overwrite the step function
# Rnadom net is useless, NetHackNet is where the action is. Thats where we do the cropping and fancy stuff
# Through a parameter to NetHackNet, we can specify the embedding dimension. That's the thing that we are trying to learn


def _format_observations(observation, keys=("glyphs", "blstats")):
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations

def run_episode(env):
    # create instance of MyAgent
    from MyAgent import MyAgent
    agent = MyAgent(env.observation_space, env.action_space)

    done = False
    episode_return = 0.0
    state = env.reset()
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return


if __name__ == '__main__':
    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")# Its the 16 cardinal points + upstairs, downstairs, chilling , reading messages, kick, eat and search
    dummy= env.reset() # The players rating and info is stored in the state array
    # The message is encoded using unicode ( the translator @ : https://www.branah.com/unicode-converter    to read the message)
    # Will be a zero array if there is no message
    # LOOK INTO WHAT SPECIALS ARE?

    print(result)
    #env.step(1)
    #env.render()
    #print(dummy)
    """
    hyper_params = {
        "seed": seeds[1],  # which seed to use
        "env": "NetHackScore-v0",  # name of the game
        "replay-buffer-size": int(10000),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 20000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold   THIS EXPLORATION STRATEGY IS GOING TO CHANGE ALOT!
        "eps-fraction": 0.6,  # fraction of num-steps
        "print-freq": 10,
    }

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    # TODO Create dqn agent
    agent = MyAgent(env.observation_space, env.action_space, replay_buffer, hyper_params["use-double-dqn"],
                     hyper_params["learning-rate"], hyper_params["batch-size"], hyper_params["discount-factor"])
    episode_rewards = []
    total_reward = 0
    the_epsilons=[]

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])


    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        # This is for exploration
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
                hyper_params["eps-end"] - hyper_params["eps-start"]
        )

        sample = random.random()
        if sample < eps_threshold:
            action = np.random.choice(agent.action_space.n)
        else:
            action = agent.act(state)
        state_prime, reward, done, _ = env.step(action)
        total_reward += reward

        replay_buffer.add(state, action, reward, state_prime, float(done))
        state= state_prime
        if done:
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset()

        if (
                t > hyper_params["learning-starts"]
                and t % hyper_params["learning-freq"] == 0
        ):
            ans=agent.optimise_td_loss()
            wandb.log({"Loss":ans, "Steps":t})
            wandb.log({"Loss":ans, "Episodes":len(episode_rewards)+1})

        if (
                t > hyper_params["learning-starts"]
                and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()
        num_episodes = len(episode_rewards)

        if (
                done
                and hyper_params["print-freq"] is not None
                and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)


            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(eps_threshold))
            print("********************************************************")
    agent.save_network()






    # TODO : Check what is the difference between glyph and character
    # Guaranteed!: The dungeon shape will always be within (21,79)
    #We dont care about the inventory since we cannot use weapons, store food, wear robes, etc
    # When we see food, we just eat it --- no questions asked!

    # The glyph id array cannot be used as is since it is not directly suitable for machine learning. They "provide some mysterious tooling to decipher the type of glyph" since most are swallow
    # The 'glpyh' state is the current text-based environment
    """
    """ # This was provided! We need to include this
    # Number of times each seed will be run
    num_runs = 10

    # Run a few episodes on each seed
    rewards = []
    for seed in seeds:
        env.seed(seed)
        seed_rewards = []
        for i in range(num_runs):
            seed_rewards.append(run_episode(env))
        rewards.append(np.mean(seed_rewards))

    # Close environment and print average reward
    env.close()
    print("Average Reward: %f" %(np.mean(rewards)))
    """
