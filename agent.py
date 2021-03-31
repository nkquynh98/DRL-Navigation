import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque
from DQN_model import DQL_network

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN_agent():
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.action_size = action_size
        self.state_size = state_size
        self.seed = random.seed(seed)

        #initilize 2 network for the agent
        self.qnetwork_local = DQL_network(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQL_network(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = LR)

        #Initilize replay memory for saving experiences
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        #Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        """Update the network"""
        #save the new experience to the memory buffer
        self.memory.add(state, action, reward, next_state, done)

        #Learn every UPDATE_EVERY time steps 
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #Check if there are enough samples for learning in the memory
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #print(state)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        #Calculate Q_next_states
        Q_next_states = self.qnetwork_target.forward(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + gamma*Q_next_states*(1-dones)
        #print(Q_next_states)
        #Calculate Q target
        #Q_target = rewards + Q_next_states.

        #Calculate Q_expected by mapping the calculated value of Q(states) to the according value of actions (index)
        Q_expected = self.qnetwork_local.forward(states).gather(1, actions)
        #print(Q_expected.shape)

        #calculate the loss
        loss = F.mse_loss(Q_expected, Q_target)
        #Reset the grad
        self.optimizer.zero_grad()
        #Backward Propagration on the loss
        loss.backward()
        #Update the parameters
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param)        
class ReplayBuffer():
    """Fixed-sized buffer to store experience for learning"""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """ Add new experience tuple to the memory """
        new_exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(new_exp)

    def sample(self):
        """Randomly choose a batch with size = batch_size from the experices memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        """Return the length of the current memory"""
        return len(self.memory) 