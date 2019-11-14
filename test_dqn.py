from tqdm import tqdm
import torch.optim as optim
from dqn import *
from train_dqn import num_TCAM, num_features, GAMMA, EPSILON, EPSILON_START, EPSILON_FINAL, EPSILON_DECAY, ACTION_SPACE, STATE_SHAPE, TARGET_FILE, rules_file, REPLAY_SIZE

# parameters with problem
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create TCAMs
tcams = [TCAM() for i in range(num_TCAM)]

# create net and target net
net = DQN(state_shape=STATE_SHAPE, action_space=ACTION_SPACE, device=DEVICE)
tgt_net = torch.load(TARGET_FILE)

buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

# create optim and environment
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
        action = agent.generate_target_decision(state)
        _ = env.step(action, rules[i])

    # total moves
    total_move = 0
    for i in range(len(tcams)):
        print("\nTCAM #{}: current rules num: {}, current move num: {}.".format(i, tcams[i].cur_num, tcams[i].move))
        total_move += tcams[i].move
    print("Total move: {}".format(total_move))


