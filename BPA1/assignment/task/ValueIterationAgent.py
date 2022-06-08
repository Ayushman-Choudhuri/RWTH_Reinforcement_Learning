from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        # self.V = ...
        self.V = {s: 0 for s in states}
        # ************

        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if len(actions) != 0:
                    arr = np.array([])
                    for a in actions:
                        p = np.array(self.mdp.getTransitionStatesAndProbs(s, a), dtype=object)
                        # print(p)
                        r = np.array(self.mdp.getReward(s, a, None), dtype=object)
                        print("reward:", r)
                        d = self.discount
                        sub = 0
                        for j in range(p.shape[0]):
                            sub += p[j][1] * (r + d * self.V[p[j][0]])
                        arr = np.append(arr, sub)
                    if arr.size != 0:
                        newV[s] = np.max(arr)
                    else:
                        newV[s] = 0.0
                else:
                    newV[s] = 0.0
            # Update value function with new estimate
            # self.V =
            self.V = newV
            # ***************
            print("self.V: ", self.V)

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        newQ = 0
        if action is not None:
            p = np.array(self.mdp.getTransitionStatesAndProbs(state, action), dtype=object)
            # print(p)
            r = np.array(self.mdp.getReward(state, action, None), dtype=object)
            d = self.discount
            sub = 0
            for j in range(p.shape[0]):
                sub += p[j][1] * (r + d * self.V[p[j][0]])
            newQ = sub
            # print("newV: ", newV)
        #
        else:
            newQ = 0.0

        return newQ
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        print(actions)
        if len(actions) < 1:
            return None
        else:
            # **********
            # TODO 2.4
            newPi = []
            for a in actions:
                q_val = self.getQValue(state, a)
                print("q value: ", q_val)
                newPi.append(q_val)
            new_Policy = actions[np.argmax(newPi)]

            return new_Policy
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
