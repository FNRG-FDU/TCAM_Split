import torch
import matplotlib.pyplot as plt
import csv
import collections
from ast import literal_eval
from enum import Enum, unique
from TCAM2 import *



class Action:
    pass


class Agent:
    pass

class Environment:
    pass


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state'])


class ExperienceBuffer:
    """
    Experience buffer class
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """
        Append an experience item
        :param experience: experience item
        :return: nothing
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample a batch from this buffer
        :param batch_size: sample size
        :return: batch: List
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, next_states


def read_rules(file_name: str):
    """
    Read rules from files
    :param file_name: file name
    :return: rule set
    """
    rules = []
    f = open(file_name)
    s = f.readlines()
    for line in s:
        ss = line.split('  ')
        rules.append(Rule(int(ss[0]), int(ss[1]), int(ss[2]), int(ss[3]), int(ss[8])))
    return rules


def generate_action_space(num_TCAM: int):
    """
    Generate action space
    :param num_TCAM: number of TCAMs
    :return: list of actions
    """
    return range(num_TCAM)

@unique
class VariableState(Enum):
    Uninitialized = 0

def fanin_init(size, fanin: float, device: torch.device = torch.device("cpu")):
    """
    Init weights
    :param size: tensor size
    :param fanin:
    :return:
    """
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v).to(device)


def printAction(action, window):
    sum_list = []
    i = 0
    while i < len(action):
        sum_list.append(sum(action[i: i + window: 1]) / window)
        i = i + window
    plt.plot(sum_list)
    plt.show()


def readDataset(path):
    data = []
    dataset = csv.reader(open(path, encoding='utf_8_sig'), delimiter=',')
    for rol in dataset:
        data.append(rol)
    data = data[1:len(data):1]
    for i in range(len(data)):
        data[i][0] = literal_eval(data[i][0])
        data[i][1] = literal_eval(data[i][1])
        data[i][3] = literal_eval(data[i][3])
        data[i][2] = float(data[i][2])
    return data


def formatnum(x, pos):
    return '$%.1f$x$10^{4}$' % (x / 10000)


def plotActionTrace(action_trace):
    for key in action_trace.keys():
        plt.plot(action_trace[key], label=str(int(key)))
    plt.xlabel("Iterations")
    plt.ylabel("Action")
    plt.title("Agent's Output with Time")
    plt.ylim((0, 100000))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
