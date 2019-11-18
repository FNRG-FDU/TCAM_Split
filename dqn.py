import torch.nn as nn
import random
from typing import List
from utils import *


@unique
class Space(Enum):
    Unlimit = 0


class DQN(nn.Module):
    def __init__(self, state_shape: tuple, action_space: List, device: torch.device):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.device = device
        self.num_features = state_shape[0]
        self.ReLU = nn.ReLU()
        self.BNs = nn.ModuleList()

        self.BNs.append(nn.BatchNorm1d(num_features=self.num_features))
        self.fc1 = nn.Linear(in_features=self.num_features, out_features=20)
        self.BNs.append(nn.BatchNorm1d(num_features=20))
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.BNs.append(nn.BatchNorm1d(num_features=20))
        self.fc3 = nn.Linear(in_features=20, out_features=len(self.action_space))

        self.init_weights(3e9)

    def init_weights(self, init_w: float):
        for bn in self.BNs:
            bn.weight.data = fanin_init(bn.weight.data.size(), init_w, device=self.device)
            bn.bias.data = fanin_init(bn.bias.data.size(), init_w, device=self.device)
            bn.running_mean.data = fanin_init(bn.running_mean.data.size(), init_w, device=self.device)
            bn.running_var.data = fanin_init(bn.running_var.data.size(), init_w, device=self.device)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), init_w, device=self.device)
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size(), init_w, device=self.device)

        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), init_w, device=self.device)
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size(), init_w, device=self.device)

        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size(), init_w, device=self.device)
        self.fc3.bias.data = fanin_init(self.fc3.bias.data.size(), init_w, device=self.device)


    def forward(self, x: torch.Tensor):
        # x = self.BNs[0](x)
        x = self.fc1(x)

        x = self.ReLU(x)

        # x = self.BNs[1](x)
        x = self.fc2(x)
        x = self.ReLU(x)

        # x = self.BNs[2](x)
        x = self.fc3(x)
        print("output: ", x)
        # print("outputs: ", outputs)
        return x


class DQNAgent(Agent):
    """
    This class is denoted as an agent used in reinforcement learning
    """

    def __init__(self, net: DQN, tgt_net: DQN, buffer: ExperienceBuffer, action_space: List, gamma: float, epsilon_start: float, epsilon: float, epsilon_final: float, epsilon_decay: float, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.gamma = gamma
        self.idx = 0

    def generate_target_decision(self, state: List):
        state_a = np.array([state], copy=False)  # make state vector become a state matrix
        state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
        self.tgt_net.eval()
        q_vals_v = self.tgt_net(state_v)  # input to network, and get output
        _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
        action_index = int(act_v.item())
        action = self.action_space[action_index]
        return action

    def generate_sample_decision(self, state: List):
        if np.random.random() < self.epsilon:
            action = random.randint(0, len(self.action_space) - 1)
            action_index = action
        else:
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = int(act_v.item())  # returns the value of this tensor as a standard Python number. This only works for tensors with one element.
        action = self.action_space[action_index]
        self.epsilon = max(self.epsilon_final, self.epsilon_start - self.idx / self.epsilon_decay)
        self.idx += 1
        return action


def calc_loss(batch, net, tgt_net, gamma: float, device: torch.device):
    states, actions, rewards, next_states = batch

    # transform each action to index(real action)
    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float).to(device)

    print("state: ", states_v)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


class DQNEnvironment(Environment):
    def __init__(self, tcams: List[TCAM]):
        super().__init__()
        self.tcams = tcams

    def step(self, action: int, rule: Rule):
        """
        execute action and get next state and reward
        :param action: the action need to execute
        :param rule: the rule need to be processed
        :return: reward
        """
        move = self.tcams[action].insert(rule)
        return -move

    def get_state(self, cur_rule: Rule):
        """
        get the state of current rule and current environment
        :param cur_rule: current rule
        :return: state
        """
        state = []
        for tcam in self.tcams:
            state.extend(tcam.get_state(cur_rule))
        state.extend()
        return state
