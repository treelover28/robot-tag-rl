from base_agent import Base_Agent
import rospy
import numpy as np


class Random_Walking_Agent(Base_Agent):
    def __init__(self, agent_type, get_agent_state_function, agent_take_action_function, get_game_information, random_action_chance):
        super(Random_Walking_Agent, self).__init__(agent_type, None, None, get_agent_state_function, agent_take_action_function, get_game_information)
        self.random_action_chance = random_action_chance
        self.agent_algorithm = "Random-Walk"

    def load_agent(self, agent_name = None):
        rospy.loginfo("Random Walking Agent does need to be loaded.")
        pass
    
    def get_policy(self, state = None, verbose = None, epsilon= None):
        rospy.loginfo("Random Walking Agent does not use a policy")
        pass

    def learn(self, epsilon = None, time_to_apply_action = None):
        rospy.loginfo("Random Walking Agent does not learn")
        pass

    def follow_policy(self, time_to_apply_action, verbose = False):
        translational_velocity, angular_velocity = self._random_walk_behavior()
        self.agent_take_action_function(self.agent_type, translational_velocity, angular_velocity, time_to_apply_action)

    def save_agent(self, agent_filename = None):
        rospy.loginfo("Random Walking Agent does not have a policy to be saved")
        pass

    def reward_function(self, state = None, verbose = None):
        rospy.loginfo("Random Walking Agent does not learn and thus does not need a reward function")
        pass
    
    def _random_walk_behavior(self):
        is_stuck = self.get_game_information("{}_stuck".format(self.agent_type))
        robot_state = self.get_agent_state_function(self.agent_type, "discrete")

        if robot_state["Front"] == "Close" and robot_state["Upper Left"] == "Too Close" and robot_state["Upper Right"] == "Too Close" and is_stuck:
            translation_speed = -0.1 
            turn_angle = -60
        elif robot_state["Front"] == "Close" and is_stuck:
            translation_speed = -0.35
            turn_angle = -60
        else:
            translation_speed = 0.15
            # 20% chance of making random turns
            if (np.random.random() < self.random_action_chance):
                turn_angle = np.random.randint(0,359)
            else:
                # just go straight else
                turn_angle = 0 
         # rospy.loginfo("translation_speed: {}, turn_angle {}".format(translation_speed, turn_angle))
        return (translation_speed, turn_angle)