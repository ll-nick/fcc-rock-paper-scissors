import numpy as np
import random

# Q-learning constants
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Length of history to consider
n_history_opponent = 2
n_history_agent = 2

# Initialize states
q_table = {}
agent_history = []
last_state = ""

def player(opponent_prev_play, opponent_history=[]):
    global q_table, epsilon, agent_history, last_state, n_history_opponent, n_history_agent

    if opponent_prev_play == "":
        # reset() # <- uncomment to restart learning with every new opponent
        agent_play = np.random.choice(["R", "P", "S"])
        agent_history.append(agent_play)
        return agent_play

    # Maintain play history per player
    opponent_history.append(opponent_prev_play)
    opponent_history = opponent_history[-n_history_opponent:]
    agent_history = agent_history[-n_history_agent:]

    # State consists of concatenated plays of the opponent and the agent
    state = get_state(opponent_history, agent_history)

    # Default to random action if not enough plays have happened yet
    if len(opponent_history) < n_history_opponent or len(agent_history) < n_history_agent:
        agent_play = np.random.choice(["R", "P", "S"])
        agent_history.append(agent_play)
        last_state = state
        return agent_play
    
    update_q_value(last_state, agent_history[-1], reward(opponent_prev_play, agent_history[-1]))

    # Choose an action using an epsilon-greedy policy
    if np.random.random() < epsilon:
        agent_play = np.random.choice(["R", "P", "S"])
    else:
        # Choose action with the highest q-value for the current state
        # If multiple actions have the same value, pick randomly
        q_table.setdefault(state, {"R": 0, "P": 0, "S": 0})
        max_value = max(q_table[state].values())
        best_actions = [key for key, value in q_table[state].items() if value == max_value]
        agent_play = random.choice(best_actions)

    agent_history.append(agent_play)
    last_state = state

    return agent_play

def reset():
    global q_table
    q_table = {}

def get_state(opponent_history, agent_history):
    return "".join(play for play in opponent_history).join(play for play in agent_history)

def update_q_value(state, agent_play, reward):
    global q_table, learning_rate, discount_factor

    q_table.setdefault(state, {"R": 0, "P": 0, "S": 0})
    old_q_value = q_table[state][agent_play]
    max_q_value = max(q_table[state].values())
    updated_q_value = (1 - learning_rate) * old_q_value + learning_rate * (reward + discount_factor * max_q_value)
    q_table[state][agent_play] = updated_q_value

def reward(opponent_play, agent_play):
    if agent_play == opponent_play:
        return 0
    elif (agent_play == "R" and opponent_play == "S") or \
            (agent_play == "P" and opponent_play == "R") or \
            (agent_play == "S" and opponent_play == "P"):
        return 1
    else:
        return -1
