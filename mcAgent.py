class MCagent:

    def __init__(self,
                 gamma: float,
                 lr: float):

        self.memory = ""
        self.gamma = gamma
        self.lr = lr

    def reset_memory(self):
        pass

    def compute_returns(self):
        pass

    def compute_first_visit_returns(self):
        pass

    def get_action(self, state):
        return action

    def update_values(self, state):
        pass

    def improve_policy(self):
        self.reset_memory()

    def reset_policy(self):
        pass
