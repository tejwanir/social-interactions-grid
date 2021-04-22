import numpy as np
from BeliefPomdp import BeliefPomdp
from Pomdp import Pomdp
from SIPomdp import SIPomdp

states = np.array([0.8, 0, 0.2, 0])
observations = np.array([0.8, 0, 0.2, 0.0 ])
actions = np.array([0, 1])

transitions = np.array([
    [[[0.05, 0.125, 0.7, 0.125],[0.125, 0.05, 0.125, 0.7]], [[0.125, 0.7, 0.125, 0.05],[0.7, 0.125, 0.05, 0.125]]],
    [[[0.125, 0.05, 0.125, 0.7],[0.05, 0.125, 0.7, 0.125]], [[0.7, 0.125, 0.05, 0.125],[0.125, 0.7, 0.125, 0.05]]],
    [[[0.7, 0.125, 0.05, 0.125],[0.125, 0.7, 0.125, 0.05]], [[0.125, 0.05, 0.125, 0.7],[0.05, 0.125, 0.7, 0.125]]],
    [[[0.125, 0.7, 0.125, 0.05],[0.7, 0.125, 0.05, 0.125]], [[0.05, 0.125, 0.7, 0.125],[0.125, 0.05, 0.125, 0.7]]]]
)

observe = np.array([
    [[0,15, 0.05, 0.075, 0.725],[0,8, 0.1, 0.03, 0.07]],
    [[0.1, 0.1, 0.6, 0.2],[0.15, 0.7, 0.075, 0.075]],
    [[0.2, 0.6, 0.1, 0.1],[0.05, 0.075, 0.725, 0.15]],
    [[0.5, 0.3, 0.1, 0.1],[0.1, 0.1, 0.2, 0.6]]]
)

reward = np.array([
    [[[-1],[50]], [[-1], [-1]]],
    [[[50],[-1]], [[-1], [-1]]],
    [[[-1],[-1]], [[50], [-1]]],
    [[[-1],[-1]], [[-1], [50]]]]
)

pomdp = Pomdp(states, observations, actions, transitions, observe, reward)

other_agent = BeliefPomdp(pomdp)
other_agent.reward = np.swapaxes(other_agent.reward,1,2)
other_agent.transitions = np.swapaxes(other_agent.transitions,1,2)
other_agent_model = np.zeros((states.shape[0], actions.shape[0])) 

print ('Before other agent model is \n', other_agent_model)
k = 3
horizon = 5
assumed_policy = np.full((states.shape[0], actions.shape[0]), 0.5)
for i in range (0, k):
    levelk = other_agent.get_opposing_policy(assumed_policy, horizon)
    other_agent_model += levelk
    if i < k-1:
        assumed_policy = BeliefPomdp(pomdp).get_opposing_policy(levelk, horizon)


other_agent_model = other_agent_model/k
print("\n\nThe inferred other agent's model is: \n", other_agent_model)
# print(other_agent_model)
# # print(np.zeros((states.shape[0], actions.shape[0])))

agent = SIPomdp(pomdp)
print()
print(agent.opt_action_selection(other_agent_model, 3))
print("\n")
