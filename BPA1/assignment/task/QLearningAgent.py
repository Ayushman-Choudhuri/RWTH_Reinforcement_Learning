import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}
        self.V = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        return self.V[state]
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        flag = 0
        actions = self.actionFunction(state)
        for i in range(len(self.Q)):
            if state == list(self.Q.keys())[i][0]:
                flag += 1
            # If state is not in self.Q, initialize Q value for state with qInitValue and state value as 0.0
        if flag == 0:
            for a in actions:
                self.V[state] = 0.0
                self.Q[state, a] = self.qInitValue
        return self.Q[(state, action)]
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.

        flag = 0
        actions = self.actionFunction(state)

        # Check if state is already in self.Q
        for i in range(len(self.Q)):
            if state == list(self.Q.keys())[i][0]:
                flag += 1

        # If state is not in self.Q, initialize Q value for state with qInitValue, state value as 0.0,
        # and return random action
        if flag == 0:
            for a in actions:
                self.V[state] = 0.0
                self.Q[state, a] = self.qInitValue
            return self.getRandomAction(state)
        # If state is already in self.Q, return greedy action
        else:
            arr = np.array([])
            for a in actions:
                arr = np.append(arr, self.Q[state, a])
            return actions[np.argmax(arr)]
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if np.random.rand() < self.epsilon:
            return self.getRandomAction(state)
        else:
            return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        flag = 0
        s_actions = self.actionFunction(state)

        # Check if state is already in self.Q
        for i in range(len(self.Q)):
            if state == list(self.Q.keys())[i][0]:
                flag += 1
        # If state is not in self.Q, initialize Q value for state with qInitValue and state value as 0.0
        if flag == 0:
            for a in s_actions:
                self.V[state] = 0.0
                self.Q[state, a] = self.qInitValue

        # Check if nextState is already in self.Q
        flag = 0
        ns_actions = self.actionFunction(nextState)
        for i in range(len(self.Q)):
            if nextState == list(self.Q.keys())[i][0]:
                flag += 1
        # If nextState is not in self.Q, initialize Q value for nextState with qInitValue and nextState value as 0.0
        if flag == 0:
            for a in ns_actions:
                self.V[nextState] = 0.0
                self.Q[nextState, a] = self.qInitValue

        # Find Q(nextState, a) for all possible actions from nextState
        arr = np.array([])
        for a in ns_actions:
            arr = np.append(arr, self.Q[nextState, a])
        if arr.size != 0:
            q_ns = np.max(arr)
        else:
            q_ns = 0.0

        # If nextState is Terminal state, set nextState values to 0.0
        if len(ns_actions) == 0:
            self.V[nextState] = 0.0

        # Update state value and q value using Q-learning update rule
        self.V[state] = self.V[state] + self.learningRate * (reward + (self.discount * self.V[nextState]) - self.V[state])
        self.Q[(state, action)] = self.Q[(state, action)] + self.learningRate * (reward + (self.discount * q_ns) - self.Q[(state, action)])
        # *********
