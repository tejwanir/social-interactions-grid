import numpy as np

class BeliefPomdp:
    
    def __init__(self, pomdp, discount = None):
        self.states = pomdp.states
        self.agent_actions = pomdp.agent_actions
        self.other_agent_actions = pomdp.other_agent_actions
        self.transitions = pomdp.transitions
        self.reward = pomdp.reward
        if discount == None:
            self.discount = 0.90
        else:
            self.discount = discount

    def get_opposing_policy(self, other_agent_policy, horizon):
        ret_policy = np.zeros((self.states.shape[0], self.agent_actions.shape[0]))
        for s,_ in enumerate(self.states):
            for u,_ in enumerate(self.agent_actions):
                for v,_ in enumerate(self.other_agent_actions):
                    ret_policy[s][u] = ret_policy[s][u] + other_agent_policy[s][v] * (self.reward[s][u][v] + self.discount * self.sum_over_trans_value(s, u, v, other_agent_policy, horizon))
            ret_policy[s] = ret_policy[s]/np.sum(ret_policy[s])
            #ret_policy[r] = np.sum(ret_policy[s])
        return ret_policy

            
    # returns the Q value of a state
    # s is the state
    def get_q_state_value(self, s, other_agent_policy, horizon):
        if horizon == 0:
            return 0
        val_list = []
        for u,_ in enumerate(self.agent_actions):
            val_sum = 0
            for v,_ in enumerate(self.other_agent_actions):
                val_sum = val_sum + other_agent_policy[s][v] *  (self.reward[s][u][v] + self.discount * self.sum_over_trans_value(s, u, v, other_agent_policy, horizon))
            val_list.append(val_sum)
        return max(val_list)


    def sum_over_trans_value(self, s, u, v, other_agent_policy, horizon):
        val = 0
        for s_prime,_ in enumerate(self.states):
            val = val + (self.transitions[s][u][v][s_prime] * self.get_q_state_value(s_prime, other_agent_policy, horizon - 1))
        return val



