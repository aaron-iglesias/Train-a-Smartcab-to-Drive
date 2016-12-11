import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, learning_rate=0.2, discount_factor=0.2):
        super(LearningAgent, self).__init__(env)
        self.color = 'red' 
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # initialize Q value to 0 for each state/action pair
        # 8 x 4 list (4 possible actions per state, 8 possible states)
        self.Q_values = np.zeros((7, 4))
        self.state_action = np.zeros((7, 4), dtype=bool)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.run = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.run += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self.calculate_state(inputs)

        # Select action according to policy
        # Chance of random action to explore state-space
        best_action = self.calculate_action()
        action = self.env.valid_actions[best_action]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        next_state = self.calculate_state(self.env.sense(self))
        next_action = self.calculate_action(next_action=True)
        self.Q_values[self.state][best_action] += self.learning_rate * (reward + self.discount_factor * self.Q_values[next_state][next_action] - self.Q_values[self.state][best_action])

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    # finds the best action from Q-values
    # actions with same Q-values have same chance via random number generator
    def calculate_action(self, next_action=False):
        if self.run <= 90 and not next_action:
            for a in range(self.state_action.shape[1]):
                if not self.state_action[self.state][a]:
                    self.state_action[self.state][a] = True
                    return a
            for s in range(self.state_action.shape[0]):
                for a in range(self.state_action.shape[1]):
                    if not self.state_action[s][a]:
                        return random.randint(0, self.state_action.shape[1] - 1)
        best_actions = [0]
        for a in range(1, self.Q_values.shape[1]):
            if self.Q_values[self.state][a] > self.Q_values[self.state][best_actions[0]]:
                best_actions = [a]
            elif self.Q_values[self.state][a] == self.Q_values[self.state][best_actions[0]]:
                best_actions.append(a)
        return random.choice(best_actions)

    # each state is distinct integer in range [0, 6]
    def calculate_state(self, inputs):
        if inputs['light'] == 'green':
            if self.next_waypoint == 'forward':
                return 0
            elif self.next_waypoint == 'right':
                return 1
            else:
                if inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'left':
                    return 2
                else:
                    return 3
        else:
            if self.next_waypoint == 'right':
                if inputs['left'] == 'forward':
                    return 4
                else:
                    return 5
            else:
                return 6

def run():
    """Run the agent for a finite number of trials."""
    e = Environment() 
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    sim = Simulator(e, update_delay=0.00, display=False)  # create simulator (uses pygame when display=True, if available)
    sim.run(n_trials=1000)  # run for a specified number of trials
    print a.Q_values

if __name__ == '__main__':
    run()