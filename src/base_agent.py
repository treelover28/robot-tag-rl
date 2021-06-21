class Base_Agent(object):
    """
    An abstract class meant to be inherited and overried by other AI agent classes.
    """
    def __init__(self, agent_type, learning_rate, discount_factor, get_agent_lidar_readings, agent_take_action_function, get_game_information):
        self.agent_type = agent_type
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.get_agent_lidar_readings = get_agent_lidar_readings
        self.agent_take_action_function = agent_take_action_function
        self.get_game_information = get_game_information

    def load_agent(self, agent_name):
        raise NotImplementedError
    
    def get_policy(self, state, verbose, epsilon):
        raise NotImplementedError

    def learn(self, epsilon, time_to_apply_action):
        raise NotImplementedError

    def follow_policy(self, time_to_apply_action, verbose):
        raise NotImplementedError

    def save_agent(self, agent_filename):
        raise NotImplementedError

    def reward_function(self, state, verbose):
        raise NotImplementedError