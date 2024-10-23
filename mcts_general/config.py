class MCTSAgentConfig:

    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.999
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.exploration_constant = 24
        self.num_simulations = 100
        self.reuse_tree = False
        self.temperature = 0
        self.do_roll_outs = True
        self.number_of_roll_outs = 1
        self.max_roll_out_depth = 20
        self.do_roll_out_steps_with_simulation_true = False
        self.use_strategies = True
        self.tau = 0.1
        self.init_decay = 0.95
