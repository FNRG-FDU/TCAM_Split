from tqdm import tqdm
import torch.optim as optim
import torch
from dqn import *

# parameters with problem
GRAIN = 16
num_TCAM = 4

# parameters with rl
GAMMA = 0.9
BATCH_SIZE = 5000

ACTION_SHAPE = 2
REPLAY_SIZE = 5000
EPSILON = 0.0
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY = 1000 #todo
LEARNING_RATE = 1e-3
SYNC_INTERVAL = 5
ACTION_SPACE = generate_action_space(num_TCAM=num_TCAM)
IN_CHANNELS = num_TCAM
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create TCAMs
tcams = [TCAM() for i in range(num_TCAM)]

# create net and target net
net = DQN(in_channels=IN_CHANNELS, grain=GRAIN, action_space=ACTION_SPACE, device=DEVICE)
tgt_net = DQN(in_channels=IN_CHANNELS, grain=GRAIN, action_space=ACTION_SPACE, device=DEVICE)
buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

# create optim and environment
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
agent = DQNAgent(net=net, tgt_net=tgt_net, buffer=buffer, action_space=ACTION_SPACE, gamma=GAMMA, epsilon_start=EPSILON_START, epsilon=EPSILON, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, device=DEVICE)
env = DQNEnvironment(tcams=tcams, grain=grain)


rules = read_rules("./data/acl1.txt")


# related
action = VariableState.Uninitialized
reward = VariableState.Uninitialized
state = VariableState.Uninitialized
idx = 0

# main function
if __name__ == "__main__":
    # deploy sfcs / handle each time slot
    for i in range(len(rules)):
        idx += 1
        state = env.get_state(rules[i])
        action = agent.generate_decision(state)
        reward = env.step(action, rules[i])
        next_state = env.get_state(rules[i])
        exp =  Experience(state=state, action=action, reward=reward, new_state=next_state)
        agent.buffer.append(exp)

        if len(agent.buffer) < REPLAY_SIZE:
            continue

        if idx % SYNC_INTERVAL == 0:
            agent.tgt_net.load_state_dict(agent.net.state_dict())

        optimizer.zero_grad()
        batch = agent.buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, agent.net, agent.tgt_net, gamma=GAMMA, action_space=ACTION_SPACE, device=DEVICE)
        loss_t.backward()
        optimizer.step()


