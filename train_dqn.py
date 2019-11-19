from tqdm import tqdm
import os
import time
import pickle
import torch.optim as optim
from dqn import *

# parameters with problem
num_TCAM = 2
num_features = num_TCAM * 1091 + 2
rules_file = "./data/fw4.txt"

# parameters with rl
LEARNING_FROM_LAST = False
GAMMA = 0.9
BATCH_SIZE = 200

SLOW_START = 500
REPLAY_SIZE = 1000
EPSILON = 0.0
EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY = 3000
LEARNING_RATE = 1e-3
SYNC_INTERVAL = 3
TRAIN_INTERVAL = 5
ACTION_SPACE = generate_action_space(num_TCAM=num_TCAM)
ACTION_LEN = len(ACTION_SPACE)
STATE_LEN = num_features
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAMPLE_FILE = "model/sample"
TARGET_FILE = "model/target"
EXP_REPLAY_FILE = "model/replay.pkl"

# create TCAMs
tcams = [TCAM() for i in range(num_TCAM)]

# create net and target net
if LEARNING_FROM_LAST:
    net = torch.load(SAMPLE_FILE)
    tgt_net = torch.load(TARGET_FILE)
    with open(EXP_REPLAY_FILE, 'rb') as f:
        buffer = pickle.load(f)  # read file and build object
else:
    net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE)
    tgt_net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE)
    for target_param, param in zip(tgt_net.parameters(), net.parameters()):
        target_param.data.copy_(param.data)
    buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

# create optim and environment
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

agent = DQNAgent(net=net, tgt_net=tgt_net, buffer=buffer, action_space=ACTION_SPACE, gamma=GAMMA, epsilon_start=EPSILON_START, epsilon=EPSILON, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, device=DEVICE)
env = DQNEnvironment(tcams=tcams)


rules = read_rules(rules_file)


# related
action = VariableState.Uninitialized
reward = VariableState.Uninitialized
state = VariableState.Uninitialized
idx = 0

# main function
if __name__ == "__main__":
    # deploy sfcs / handle each time slot
    for i in tqdm(range(len(rules))):
        idx += 1
        state = env.get_state(rules[i])
        action = agent.generate_sample_decision(state)

        if i < SLOW_START:
            if rules[i].height > rules[i].wide:
                action = 0
            else:
                action = 1

        reward = env.step(action, rules[i])
        next_state = env.get_state(rules[i])
        exp =  Experience(state=state, action=action, reward=reward, new_state=next_state)
        agent.buffer.append(exp)

        if len(agent.buffer) < REPLAY_SIZE:
            continue

        if i == REPLAY_SIZE:
            total_move = 0
            for i in range(len(tcams)):
                print("\nTCAM #{}: current rules num: {}, current move num: {}.".format(i, tcams[i].cur_num, tcams[i].move))
                total_move += tcams[i].move
            print("Total move: {}".format(total_move))

        if idx % SYNC_INTERVAL == 0:
            agent.tgt_net.load_state_dict(agent.net.state_dict())

        if idx % TRAIN_INTERVAL == 0:
            optimizer.zero_grad()
            batch = agent.buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, agent.net, agent.tgt_net, gamma=GAMMA, device=DEVICE)
            loss_t.backward()
            optimizer.step()

    # save model
    torch.save(agent.net, SAMPLE_FILE)
    torch.save(agent.tgt_net, TARGET_FILE)
    with open(EXP_REPLAY_FILE, 'wb') as f: # open file with write-mode
        model_string = pickle.dump(agent.buffer, f) # serialize and save object

    # total moves
    total_move = 0
    for i in range(len(tcams)):
        print("\nTCAM #{}: current rules num: {}, current move num: {}.".format(i, tcams[i].cur_num, tcams[i].move))
        total_move += tcams[i].move
    print("Total move: {}".format(total_move))
    time.sleep(10)
    os.system("/usr/bin/python3 -u /home/user/projects/TCAM_Split/train_dqn.py")
