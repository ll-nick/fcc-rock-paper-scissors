import numpy as np

# Constants for Q-learning
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Length of history to consider
n = 3

# Initialize the Q-table
q_table = {}
agent_prev_play = ""

def reset():
    global q_table
    q_table = {}

def player(opponent_prev_play, opponent_history=[]):
    global q_table, epsilon, agent_prev_play

    if opponent_prev_play == "":
        reset()
        agent_play = np.random.choice(["R", "P", "S"])
        agent_prev_play = agent_play
        return agent_play

    opponent_history.append(opponent_prev_play)
    last_state = opponent_history[-n-1:-1]
    current_state = opponent_history[-n:]

    if len(opponent_history) < n + 1:
        agent_play = np.random.choice(["R", "P", "S"])
        agent_prev_play = agent_play
        return agent_play

    update_q_value(last_state, agent_prev_play, reward(opponent_prev_play, agent_prev_play))

    # Choose an action using an epsilon-greedy policy
    if np.random.random() < epsilon:
        agent_play = np.random.choice(["R", "P", "S"])
    else:
        state = "".join(play for play in current_state)
        q_table.setdefault(state, {"R": 0, "P": 0, "S": 0})
        agent_play = max(q_table[state], key=q_table[state].get)

    agent_prev_play = agent_play

    return agent_play

def update_q_value(opponent_history, agent_play, reward):
    global q_table, learning_rate, discount_factor

    state = "".join(play for play in opponent_history)
    q_table.setdefault(state, {"R": 0, "P": 0, "S": 0})
    old_q_value = q_table[state][agent_play]
    max_q_value = get_max_q_value(state)
    updated_q_value = (1 - learning_rate) * old_q_value + learning_rate * (reward + discount_factor * max_q_value)
    q_table[state][agent_play] = updated_q_value

def get_max_q_value(state):
    global q_table
    max_q_value = float('-inf')  # Initialize to negative infinity
    
    # Iterate through all possible actions in the state's Q-value dictionary
    for action, q_value in q_table[state].items():
        if q_value > max_q_value:
            max_q_value = q_value
    
    return max_q_value

def reward(opponent_play, agent_play):
    if agent_play == opponent_play:
        return 0
    elif (agent_play == "R" and opponent_play == "S") or \
            (agent_play == "P" and opponent_play == "R") or \
            (agent_play == "S" and opponent_play == "P"):
        return 1
    else:
        return -1