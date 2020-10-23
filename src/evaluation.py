import numpy as np
import gym
import nle
import random
from MyAgent import MyAgent, DQN
from replay_buffer import ReplayBuffer
import torch
import numpy as np
import random
from torchsummary import summary

np.set_printoptions(threshold=np.inf,linewidth=100000)
# Eventually, look into clipping the rewards (The justifacation being stated in the DeepMind DQN paper)
# use their Resetting environment to be trained indefinitely... Otherwise, it will terminate after every episode --> But this is a normal gym environment
# We're going to have to use other things as input as well because our "local view" around the agent is usually small and typically, neural networks would need larger inputs than that to be useful
#

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

def Crop(observation, stacked_world, radius = 5):
    half_radius = np.int(radius/2)
    stats = observation['blstats']
    x = stats[0]
    y = stats[1]
    return stacked_world[:,y-half_radius:y+half_radius+1,x-half_radius:x+half_radius+1]


if __name__ == '__main__':
    # Seed
    seeds = [1,2,3,4,5]

    hyper_params = {
        "seed": 42,  # which seed to use
        "replay-buffer-size": int(10000),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 20000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.6,  # fraction of num-steps
        "print-freq": 10,
        "crop_dim" : 5
    }

    # LOOK INTO WHAT SPECIALS ARE?

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    env = gym.make("NetHackScore-v0")
    # Dont change the setup of the game (Options isnt even a member!)
    # The world is zeros everywhere and is only populated where our world actually is
    # From empirical evidence, the reward function varies dramatically, its typically between 0 and -0.01( which is used as a penalty)
    # For positive rewards, it ranges from 2 to 20, so its not a good idea to clip the rewards(I think it should be sensible to maintain that variability)
    # At the moment, specials is just a zeroes array???
    # Theres definitely a difference between glyphs and chars - glyphs is zero when undefined and contains large numbers. char is 32 where undefined and has small numbers
    # The architecture of the network will require some thought
    # I've read through the Pong wrappers and I dont think we need any of them
    # Im starting with their naive epsilon strategy, will change later to be fancy

    test= env.reset()
    my_x = test['blstats'][0]
    my_y = test['blstats'][1]
    env.render()
    env.seed(hyper_params["seed"])
    # Remember that the size of the world is 21 x79
    #print(type(test["glyphs"]))
    # print(test["glyphs"])
    #print(test['chars'])



    # // This block of code is pretty unnecessary at the moment, I just have it here for debugginf purposes... It's used properly in the for-loop below
    # Below stacks the glyph, colors and chars to form a map... will have the depth channel first
    stacked_version = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(test['glyphs']),0),torch.unsqueeze(torch.from_numpy(test['colors']),0))),torch.unsqueeze(torch.from_numpy(test['chars']),0)))
    #print(stacked_version.shape)
    output = Crop(test,stacked_version,hyper_params["crop_dim"]) # One thing that is quite important and you could look into is, if our agent is close to a corner, the cropping function wont return the correct size... To fix this issue, we can pad each "slice of the map" with the
                                        # appropriate placeholder (0 for example in the case of glyphs, 32 in chars, etc)
    # For whatever it's worth, I'm creating a padded version of the world to size 79x79 incase we decide to also pass the whole world as input
    padded_world = np.zeros((3,79,79))
    padded_world[:,29:50,:] = stacked_version


    """
    while(True):
        action = random.randint(0,22)
        state_prime, reward,done,_ = env.step(action)
        #env.render()
        if(done):
            break
    """

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    agent = MyAgent(np.zeros((3,79,79)), env.action_space, replay_buffer, hyper_params["use-double-dqn"],
                    hyper_params["learning-rate"], hyper_params["batch-size"], hyper_params["discount-factor"])

    episode_rewards = []
    total_reward = 0

    the_epsilons=[]

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])

    state = env.reset()
    padded_world = np.zeros((3,79,79))
    new_padded_world = np.zeros((3,79,79))


    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
                hyper_params["eps-end"] - hyper_params["eps-start"]
        )

        stacked_version = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(state['glyphs']),0),torch.unsqueeze(torch.from_numpy(state['colors']),0))),torch.unsqueeze(torch.from_numpy(state['chars']),0)))
        output = Crop(state,stacked_version,hyper_params["crop_dim"])
        padded_world[:,29:50,:] = stacked_version
        padded_world = torch.tensor(padded_world)

        sample = random.random()
        if sample < eps_threshold:
            action = np.random.choice(agent.action_space.n)
        else:
            #print(torch.squeeze(padded_world,0))
            action = agent.act(torch.unsqueeze(padded_world,0))
        state_prime, reward, done, _ = env.step(action)
        env.render()
        new_stacked_version = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(state_prime['glyphs']),0),torch.unsqueeze(torch.from_numpy(state_prime['colors']),0))),torch.unsqueeze(torch.from_numpy(state_prime['chars']),0)))
        new_padded_world[:,29:50,:] = new_stacked_version
        new_padded_world = torch.tensor(new_padded_world)

        total_reward += reward

        replay_buffer.add(padded_world, action, reward, new_padded_world, float(done))
        state= state_prime
        if done:
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset()
        #print(t)

        if (
                t > hyper_params["learning-starts"]
                and t % hyper_params["learning-freq"] == 0
        ):
            ans=agent.optimise_td_loss()

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
