import torch.nn as nn
import random
from typing import List
from utils import *

@unique
class Space(Enum):
    Unlimit = 0


class DQN(nn.Module):
    def __init__(self, in_channels: int, grain: int, action_space: List, device: torch.device):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.device = device
        self.in_channels = in_channels
        self.grain = grain

        convs = []
        out_channels = in_channels + 1
        while grain > 1:
            grain = (grain - 1) // 2
            convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3))
            convs.append(nn.ReLU())
            in_channels = out_channels
            out_channels = out_channels + 1
        self.convs = nn.Sequential(*convs)

        self.init_weights(3e2)

    def init_weights(self, init_w):
        for layer in self.convs:
            layer.weight.data = fanin_init(layer.weight.data.size(), init_w, device=self.device)
            layer.bias.data = fanin_init(layer.bias.data.size(), init_w, device=self.device)

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        print(x)

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

    def generate_decision(self, state: List):
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




def calc_loss(batch, net, tgt_net, gamma: float, action_space: List, device: torch.device):
    states, actions, rewards, next_states = batch

    # transform each action to index(real action)

    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


class DQNEnvironment(Environment):
    def __init__(self, tcams: List[TCAM], grain: int):
        super().__init__()
        self.tcams = tcams
        self.grain = grain

    def step(self, action: int, rule: Rule):
        """
        execute action and get next state and reward
        :param action: the action need to execute
        :param rule: the rule need to be processed
        :return: reward
        """
        return -self.tcams[action].insert(rule)


    def get_state(self, cur_rule: Rule):
        """
        get the state of current rule and current environment
        :param cur_rule: current rule
        :return: state
        """
        state = []
        for tcam in self.tcams:
            state.append(self.focus(tcam.overlap_metrix, cur_rule))
        return state

    def focus(self, overlap_metrix: List, rule: Rule):
        """
        focus the overlap metrix related to this rule
        :param overlap_metrix: overlap metrix
        :param rule: rule
        :return: metrix processed
        """
        rule_metrix = np.zeros((self.grain, self.grain))
        side = pow(2, 32) / self.grain
        left = math.floor(rule.src[0] / side)
        right = math.floor(rule.src[1] / side)
        down = math.floor(rule.dst[0] / side)
        up = math.floor(rule.dst[1] / side)

        for i in range(left, right + 1):
            for j in range(down, up + 1):
                rule_metrix[i][j] = 1

        for i in range(len(overlap_metrix)):
            for j in range(len(overlap_metrix[i])):
                overlap_metrix[i][j] = 0 if rule_metrix[i][j] == 0 else overlap_metrix[i][j]

        return overlap_metrix

