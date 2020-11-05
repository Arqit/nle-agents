import gym
import nle
import random
from MyAgent import MyAgent, DQN
from replay_buffer import PrioritizedReplayBuffer
import torch
import numpy as np
import random
from torchsummary import summary


device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

# how will the network know where we are... Ignore for now and expect the best! d(._.)b
# Get the noisy net to work so that we stop getting stuck when we loop from getting the same input and producing the same answer

np.set_printoptions(threshold=np.inf, linewidth=100000)


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
        (new_state, reward, done, _) = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return


def Crop(observation, stacked_world, radius=5):
    half_radius = np.int(radius / 2)
    stats = observation['blstats']
    x = stats[0]
    y = stats[1]  # I'm adding the 29 because I'm now passing the padded world which pads above and below... To cut-pass the top padding, I have to add 29 to the row
    return stacked_world[:, 29 + y - half_radius:29 + y + half_radius + 1, x - half_radius:x + half_radius + 1]  # Fix this so that we pad to the correct size when we close to the edge!


if __name__ == '__main__':

    # Seed

    seeds = [1, 2, 3, 4, 5]

    hyper_params = {  # Tinker around with these
        'seed': 42,  # which seed to use
        'replay-buffer-size': int(1000),  # replay buffer size
        'replay-batch-size': int(32), # Batch size when sampling the replay buffer
        'learning-rate': 1e-4, # learning rate for Adam optimizer
        'discount-factor': 0.99,# discount factor
        'num-steps': int(1e6), # total number of steps to run the environment for
        'batch-size': 256, # number of transitions to optimize at the same time
        'learning-starts': 1000, # number of steps before learning starts
        'learning-freq': 5, # number of iterations between every optimization step
        'use-double-dqn': True, # use double deep Q-learning
        'target-update-freq': 1000, # number of iterations between every target network update
        'eps-start': 1.0, # e-greedy start threshold
        'eps-end': 0.01, # e-greedy end threshold
        'eps-fraction': 0.6,  # fraction of num-steps
        'print-freq': 10,
        'crop_dim': 5,
        'alpha': 0.2,
        'beta': 0.6,
        'prior_eps': 1e-6
        }

    # LOOK INTO WHAT SPECIALS ARE?

    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    env = gym.make("NetHackScore-v0") # If its automatically picking up gold, then autopickup must be enabled for everything


    print(env.__dict__)
    test = env.reset()
    my_x = test['blstats'][0]
    my_y = test['blstats'][1]
    env.seed(hyper_params['seed'])


    # // This block of code is pretty unnecessary at the moment, I just have it here for debugginf purposes... It's used properly in the for-loop below
    # Below stacks the glyph, colors and chars to form a map... will have the depth channel first
    stacked_version = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(test['glyphs']), 0),
                   torch.unsqueeze(torch.from_numpy(test['colors']), 0))),
                   torch.unsqueeze(torch.from_numpy(test['chars']), 0)))



    replay_buffer = PrioritizedReplayBuffer(hyper_params['replay-buffer-size'],batch_size= hyper_params['replay-batch-size'],alpha = hyper_params['alpha'])
    agent = MyAgent(
        np.zeros((3, 79, 79)), # assuming that we are taking the world as input
        env.action_space,
        replay_buffer,
        hyper_params['use-double-dqn'],
        hyper_params['learning-rate'],
        hyper_params['batch-size'],
        hyper_params['discount-factor'],
        hyper_params['beta'],
        hyper_params['prior_eps']
        )

    episode_rewards = []
    total_reward = 0

    the_epsilons = []

    eps_timesteps = hyper_params['eps-fraction'] * float(hyper_params['num-steps'])

    state = env.reset()
    padded_world = np.zeros((3, 79, 79))
    new_padded_world = np.zeros((3, 79, 79))

    for t in range(hyper_params['num-steps']):

        stacked_version = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(state['glyphs']), 0),
                      torch.unsqueeze(torch.from_numpy(state['colors']), 0))),
                      torch.unsqueeze(torch.from_numpy(state['chars']),0)))
        padded_world[:, 29:50, :] = stacked_version
        output = Crop(state, padded_world, hyper_params['crop_dim'])
        padded_world = torch.tensor(padded_world)

        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
                hyper_params["eps-end"] - hyper_params["eps-start"]
        )

        if random.random() < eps_threshold:  # Will presumably be replaced by the Noisy Layer stuff
            action = np.random.choice(agent.action_space.n)
        else:
            action = agent.act(torch.unsqueeze(padded_world, 0)).item()

        if action == 21:  # The eating 'macro' which attempts to handle the food selection issue (the developers need to get their act together)

            # state_prime, reward, done, _ = env.step(action)
            # the_message = [chr(a) for a in state_prime["message"]]
            # the_message = ''.join(the_message)
            # print("Im here!")
            # print(the_message)
            # if 'There' in the_message:
            #     action = ord('y')
            # else:
            #     options = the_message[the_message.find('['):the_message.find(" ",the_message.find('['))]
            #     action = ord(options[np.random.choice(len(options))])
            #     print("Eating Menu"+str(action))

            action = 3  # This is just a placeholder so that I can continue developing
        (state_prime, reward, done, _) = env.step(action)
        env.render()
        new_stacked_version = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(state_prime['glyphs']), 0),
                      torch.unsqueeze(torch.from_numpy(state_prime['colors']), 0))),
                      torch.unsqueeze(torch.from_numpy(state_prime['chars']), 0)))
        new_padded_world[:, 29:50, :] = new_stacked_version # Insert the world into our newly allocated square world
        new_padded_world = torch.tensor(new_padded_world)

        total_reward += reward

        replay_buffer.add(padded_world, action, reward,new_padded_world, float(done))
        state = state_prime
        fraction = min(t / hyper_params['num-steps'], 1.0)
        agent.beta = agent.beta + fraction * (1.0 - agent.beta)
        if done:
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset()

        if t > hyper_params['learning-starts'] and t \
            % hyper_params['learning-freq'] == 0:
            ans = agent.optimise_td_loss()

        if t > hyper_params['learning-starts'] and t \
            % hyper_params['target-update-freq'] == 0:
            agent.update_target_network()
        num_episodes = len(episode_rewards)

        if done and hyper_params['print-freq'] is not None \
            and len(episode_rewards) % hyper_params['print-freq'] == 0:
            mean_100ep_reward = \
                round(np.mean(episode_rewards[-101:-1]), 1)

            print('********************************************************')
            print('steps: {}'.format(t))
            print('episodes: {}'.format(num_episodes))
            print('mean 100 episode reward: {}'.format(mean_100ep_reward))
            print('% time spent exploring: {}'.format(eps_threshold))
            print('********************************************************')
    agent.save_network()
