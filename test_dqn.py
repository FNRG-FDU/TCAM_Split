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
IN_CHANNELS = num_TCAM + 1
DEVICE = torch.device("cuda")

# create decision maker(agent) & optimizer & environment
net = DQN(device=DEVICE)
tgt_net = DQN(device=DEVICE)
buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
env = DQNEnvironment()


rules = read_rules("./data/acl1.txt")
tcams = [TCAM() for i in range(num_TCAM)]




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
        state = env.get_state(model, i)
        decision = deploy_sfc_item(model, i, decision_maker, cur_time, state, test_env)
                action = DQNAction(decision.active_server, decision.standby_server).get_action()
                reward = env.get_reward(model, i, decision, test_env)
                next_state = env.get_state(model, i)

                exp =  Experience(state=state, action=action, reward=reward, new_state=next_state)
                decision_maker.buffer.append(exp)

                if len(decision_maker.buffer) < REPLAY_SIZE:
                    continue

                if idx % SYNC_INTERVAL == 0:
                    decision_maker.tgt_net.load_state_dict(decision_maker.net.state_dict())

                optimizer.zero_grad()
                batch = decision_maker.buffer.sample(BATCH_SIZE)
                loss_t = calc_loss(batch, decision_maker.net, decision_maker.tgt_net, gamma=GAMMA, action_space=ACTION_SPACE, device=DEVICE)
                loss_t.backward()
                optimizer.step()

    Monitor.print_log()

    # model.print_start_and_down()

    print(model.calculate_fail_rate())

    print(model.calculate_accept_rate())
