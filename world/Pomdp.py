class Pomdp:
  def __init__(self, states=None, observations=None, actions=None, transitions=None, observe=None, reward=None, discount=None):
    self.states = states
    self.observations = observations
    self.agent_actions = actions
    self.other_agent_actions = actions
    self.transitions = transitions
    self.observe = observe
    self.reward = reward
    self.discount = discount
