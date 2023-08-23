import numpy as np

# Constants for Q-learning
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Initialize the Q-table
q_table = np.zeros((3, 3))
agent_prev_play = ""
opponent_prev_prev_play = ""

def reset():
    global q_table
    q_table = np.zeros((3, 3))

play_to_num = {"R": 0, "P": 1, "S": 2}
num_to_play = {0: "R", 1: "P", 2: "S"}

def player(opponent_prev_play, opponent_history=[]):
    global q_table, epsilon, agent_prev_play, opponent_prev_prev_play

    if opponent_prev_play == "":
        reset()
        agent_play = np.random.choice([0, 1, 2])
        return num_to_play[agent_play]

    if opponent_prev_prev_play == "":
        agent_play = np.random.choice([0, 1, 2])
        agent_prev_play = agent_play
        opponent_prev_prev_play = play_to_num[opponent_prev_play]
        return num_to_play[agent_play]

    update_q_value(opponent_prev_prev_play, agent_prev_play, reward(play_to_num[opponent_prev_play], agent_prev_play))

    current_state = play_to_num[opponent_prev_play]

    # Choose an action using an epsilon-greedy policy
    if np.random.random() < epsilon:
        agent_play = np.random.choice([0, 1, 2])
    else:
        agent_play = np.argmax(q_table[current_state, :])

    opponent_prev_prev_play = play_to_num[opponent_prev_play]
    agent_prev_play = agent_play

    return num_to_play[agent_play]

def update_q_value(opponent_prev_play, agent_play, reward):
    global q_table, learning_rate, discount_factor

    old_q_value = q_table[opponent_prev_play][agent_play]
    best_next_q_value = np.argmax(q_table[agent_play, :])
    new_q_value = (1 - learning_rate) * old_q_value + learning_rate * (reward + discount_factor * best_next_q_value)
    q_table[opponent_prev_play][agent_play] = new_q_value

def reward(opponent_play, agent_play):
    agent_play_str = num_to_play[agent_play]
    opponent_play_str = num_to_play[opponent_play]
    if agent_play_str == opponent_play_str:
        return 0
    elif (agent_play_str == "R" and opponent_play_str == "S") or \
            (agent_play_str == "P" and opponent_play_str == "R") or \
            (agent_play_str == "S" and opponent_play_str == "P"):
        return 1
    else:
        return -1