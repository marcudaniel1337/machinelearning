import random

class QLearningAgent:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent.

        Args:
            states (list): All possible states.
            actions (list): All possible actions.
            learning_rate (float): How quickly the agent updates Q-values.
            discount_factor (float): Importance of future rewards.
            epsilon (float): Exploration rate (chance to pick random action).
        """
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table as dictionary: Q[state][action] = value
        self.Q = {state: {action: 0.0 for action in actions} for state in states}

    def choose_action(self, state):
        """
        Choose action based on epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            # Explore: pick a random action
            return random.choice(self.actions)
        else:
            # Exploit: pick best action from Q-table
            q_values = self.Q[state]
            max_q = max(q_values.values())
            # In case multiple actions have same max Q-value, randomly choose among them
            best_actions = [action for action, value in q_values.items() if value == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning update rule.

        Q(s,a) = Q(s,a) + lr * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        max_next_q = max(self.Q[next_state].values())
        current_q = self.Q[state][action]

        # Compute updated Q-value
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        # Update Q-table
        self.Q[state][action] = new_q

if __name__ == "__main__":
    # Example: simple environment with 3 states and 2 actions
    states = ['A', 'B', 'C']
    actions = ['left', 'right']

    agent = QLearningAgent(states, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.2)

    # Simulate some training episodes
    for episode in range(100):
        state = random.choice(states)  # start randomly
        for step in range(10):
            action = agent.choose_action(state)
            # Simulate environment response: next_state and reward
            if state == 'A':
                next_state = 'B' if action == 'right' else 'C'
                reward = 1 if next_state == 'B' else 0
            elif state == 'B':
                next_state = 'C' if action == 'right' else 'A'
                reward = 1 if next_state == 'C' else 0
            else:
                next_state = 'A' if action == 'right' else 'B'
                reward = 1 if next_state == 'A' else 0

            agent.update(state, action, reward, next_state)
            state = next_state

    # After training, print Q-values
    print("Learned Q-values:")
    for state in states:
        print(f"State {state}:")
        for action in actions:
            print(f"  Action {action}: {agent.Q[state][action]:.2f}")
