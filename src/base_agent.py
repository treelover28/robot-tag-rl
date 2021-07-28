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
        '''
        This method takes in the filename of the saved agent and loads the agent's information/Q-Tables/Network-Weights.
        '''
        raise NotImplementedError
    
    def get_policy(self, state, verbose, epsilon):
        '''
        This method either gets the best action from the agent's current policy given its current state (exploit) OR returns a random action (explore). 
        The exploration-exploitation trade-off is controlled by the epislon
        '''
        raise NotImplementedError

    def learn(self, epsilon, time_to_apply_action):
        '''
        The method takes one learning step using the agent's learning algorithm (Q-Table update or DQN's Gradient Descent Step) 
        '''
        raise NotImplementedError

    def follow_policy(self, time_to_apply_action, verbose):
        '''
        This method strictly exploits the policy. Returns the best action from the agent's current policy given its current state
        '''
        raise NotImplementedError

    def save_agent(self, agent_filename):
        '''
        This method saves the agent using Pickle. Takes in filename to be saved under.
        '''
        raise NotImplementedError

    def reward_function(self, state, verbose):
        '''
        Each agent implements its own reward function based on whether its state is continuous or discete.
        '''
        raise NotImplementedError