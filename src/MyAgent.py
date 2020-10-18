from AbstractAgent import AbstractAgent
import torch.nn as nn
import torch
import numpy as np

# We can spend the whole of Tuesday,ensuring that we thoroughly understand what exactly is going on in their model--> This will then help when we do the DQN approach (hopefully friday)
# We'll also need to include the ResetEnvironment in ours to be able to train successively for many rounds (should pay attention to how we maintain state between episodes where necessary)
# Our act function will definitely be different to theirs
# We dont need their learn function
# We'll probably take some inspiration from their train function
# I think that I've extracted all the important things that we need to be working with (besides for the ResetEnv that can be directly copied over)

def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)

# This is the thing we will call to crop the world as desired... This needs some looking into
class Crop(nn.Module):
    def __init__ (self, height,width,target_size):
        super(Crop,self).__init__()

        self.width = width
        self.height = height
        # I have no clue what does below do (as well as the 'register_buffer stuff')
        # Expand is to just stretch the input over a different shape (presumably by repeating values in a nice way)
        # Whats happening with the dividing?
        width_grid = _step_to_range(2 / (self.width - 1), self.target_size)[None, :].expand(self.height_target, -1) # -1 means leave the dimension unchanged
        height_grid = _step_to_range(2 / (self.height - 1), self.target_size)[:, None].expand(-1, self.width_target)
        # From what I read, buffers are weights and biases that we need but are not trained by an optimizer (but we want to be able to save the weights)... It is not treated as a module of a model
        # Whats the point of the weights if we're not going to train them to do useful stuff
        # I dont see why are we working with weights here... I expect to just do index slicing of the world around the agent?

        # Check where will this be used?
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs,coordinates):
        # Check what are they treating as the input
         """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W] ( The world) Whats the B for?... Maybe they stacking like us?) depth could be the glyphs, colors and chars
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        print(inputs.shape)
        print(coordinates)

        inputs = inputs[:,Nones,:,:].float() # Know clue why are we doing this
        x = coordinates[:,0]# Why are we slicing weirdly? Why not just coordinates[0]?
        y = coordinates[:,1]

        # No clue what is this supposed to achieve
        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)
        # Look into what is this grid?
        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )

class DQN(nn.Module):
    """
    Insert description here
    """

    def __init__(self,observation,num_actions,embedding_dim=32,crop_dim=9,num_layers=5):
        super.__init__()
        """Parameter explanation
        observation:We will index this dictionary to extract useful information
        num_actions: How many actions are we allowed to take
        embedding_size: Size of the embedding
        crop_dim: dimensions around that we crop around the agent
        num_layers: number of layers the agent will have

        """

        self.map_size = observation['glyph'].shape
        self.blstats_size = observation['blstats'].shape[0]
        # glpyhs, colors and chars all have the same size...
        # Will now only need to look into how/if messages are handled
        self.num_actions num_actions
        self.H = self.map_size[0]
        self.W = self.map_size[1]

        self.embed_dim = embedding_dim
        self.h_size = 512 # This is the width of the network (number of nodes in each layer)


        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim)
        self.embed = nn.Embedding(nethack.MAX_GLYPHS,self.embed_dim) # This needs to be trained!
        # Each word (glyph in our case) will be represented by a vector embedding with size vector embedding
        """
        Embedding: just creates a Lookup Table, to get the word embedding given a word index.
        I get the feeling that this is something like a mapper /translator
        """

        # Which network is this for?
        # Check if all 3 input subnetworks have the same embedding dimension)
        # These are the hyperparameters of the network
        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1) # How and why?
        out_channels = [M] * (L - 1) + [Y] # How and why?

        def interleave(xs,ys): #????
            return [val for pair in zip(xs, ys) for val in pair]

        # We now define the structure of the network ( which one - there are many networks that we are dealing with)
        # For below, how and why are they indexing a scalar? in_channels is defined above as a scalar?
        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # nn.ELU applies the elementwise function:
        # ELU(x) = max(0,x) + min(0, alpha *(exp(x)-1))--> Why is this necessary?

        # CNN crop model. --> Structure is identical to the previously defined model
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]


        # This interleave function seems to be extremely important
        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # Finally some resemblence!!!
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        # Network to embed the blstats... what exactly does this mean?

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        # Look into what is this used for? -> Check if we will be needing this
        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        """
        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)
        """

        # They are using a policy gradient method!
        """
        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)
        """

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple() # Empty!
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x): # What is this!
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state):
        # Time seems to be an important aspect (not sure if this is solely related to the LSTM?)
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]
        print(glyphs.shape)

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()

        # Check what exactly is blstats stored as, to understand the way that they are indexing
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # print("crop", crop)
        # print("at_xy", glyphs[:, coordinates[:, 1].long(), coordinates[:, 0].long()])

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep) # This is how we are collecting the different representations, this is what we need

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)
        # Now, reps contains the blstats rep, the crop rep and now the glyph_rep

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)# --> Check if we will be needing the fc
        """

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st
        """
        # Looking at the context below, I reckon that core_output should be the "ultimate" input to *our* model

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)


        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )
        """




class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space,replay_buffer, lr, batch_size, gamma):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.Q= DQN(self.observation_space,self.action_space)
        self.Q.cuda()
        self.Q_hat = DQN(self.observation_space,self.action_space)
        self.Q_hat.cuda()
        self.optimizer = torch.optim.Adam(self.Q.parameters(),lr=self.lr)

    def optimize_td_loss(self):
        samples = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(samples[0]).type(torch.cuda.FloatTensor)
        next_states = torch.tensor(samples[3]).type(torch.cuda.FloatTensor)
        actions = torch.tensor(samples[1]).type(torch.cuda.LongTensor)
        rewards = torch.tensor(samples[2]).type(torch.cuda.FloatTensor)
        done = torch.tensor(samples[4]).type(torch.cuda.LongTensor)

        actual_Q = self.Q(states).gather(1,actions.unsqueeze(-1)).squeeze(-1)
        Q_primes = self.Q_hat(next_states).max(1)[0]
        Q_primes[done] = 0.0
        Q_primes = Q_primes.detach()
        predicted_Q_values = Q_primes * self.gamma + rewards

        loss_t = nn.MSELoss()(actual_Q, predicted_Q_values)
        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()

        return loss_t.item()
        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))


    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """

        # TODO update target_network parameters with policy_network parameters
        self.Q_hat.load_state_dict(self.Q.state_dict())
        # raise NotImplementedError

    def save_network(self):
        torch.save(self.Q.state_dict(),"The_weights.pth")

    def load_network_weights(self):
        self.Q.load_state_dict(torch.load("The_weights.pth"))

    def act(self, observation):
        # Perform processing to observation

        # TODO Select action greedily from the Q-network given the state
        the_state = torch.unsqueeze(torch.from_numpy(np.array(state)), 0)
        the_state = the_state.type(torch.cuda.FloatTensor)
        the_answer = self.Q.forward(the_state).cpu()
        return torch.argmax(the_answer)
