import os
import rospy
import cPickle as pickle
import numpy as np 
from mischallenous_functions import _get_permutations
from activation_functions import sigmoid
from base_agent import Base_Agent

# from robot_tag import random_walk_behavior

class Simple_Q_Learning_Agent(Base_Agent):
    def __init__(self, agent_type, learning_rate, discount_factor, substates_ratings_list, actions_list, get_agent_state_function, agent_take_action_function, get_game_information):
        # call Agent's constructor
        super(Simple_Q_Learning_Agent, self).__init__(agent_type, learning_rate, discount_factor, get_agent_state_function, agent_take_action_function, get_game_information)

        self.agent_algorithm = "Q-Learning"
        if (not os.path.isfile("q_table_{}.txt".format(self.agent_type))):
            # if existing Q-table is not found, create a new fresh one
            rospy.loginfo("Created and initialized a new Q_Table for {}".format(self.agent_type))
            self.q_table = self.create_q_table(substates_ratings_list, actions_list)
        
        with open("q_table_{}.txt".format(self.agent_type), "rb") as q_table_file:
            self.q_table = pickle.load(q_table_file)
        
        
        
    def create_q_table(self, substates_rating_list, actions_list):
        """
        Generate a Q-Table in form of a dictionary where the key is a unique state in all O(m^n) possible states from resulting
        from n possible directions and each direction has m possible distance ratings. The value associated with each key (state)
        is a tuple consisting of (another tuple containing n q-values associated with n actions for each state, and a boolean to tell robot 
        whether to reverse or not). 

        The q-values here are manually defined (prior expert knowledge) where states are logically broken down into common scenarios a robot may face 
        while wall-following. Each common scenario has a desirable action that could address the scenario, and thus the associated q-values of such action
        is a lot higher than the q-values for other neutral actions.
    
        """

        q_table = {}
        all_states = []
        _get_permutations(0, substates_rating_list, list(), len(substates_rating_list), all_states)
        
        all_actions = []
        _get_permutations(0, actions_list ,list(), len(actions_list), all_actions)

        for state in all_states:
            # the order of the sub-states in the list is
            # front_rating, upper_left_rating, upper_right_rating, lower_left_rating, lower_right_rating, opponent_rating
            # convert state (originally a list) to a tuple which is hashable 
            # and could be stored in the Q_table which is a dictionary
            state_tuple = tuple(state)
            # initialize the q-value for each action to be 0
            q_values = {}
            for action in all_actions:
                q_values[tuple(action)] = 0

            # each state has n q-values associated with n actions that can be done, plus
            q_table[state_tuple] = q_values
        
        # save Q-table as an external file
        with open("q_table_{}.txt".format(self.agent_type), "w") as q_table_file:
            q_table_file.seek(0)
            q_table_file.write(pickle.dumps(q_table)) 

    def load_agent(self,q_table_name):
        if (not os.path.isfile(q_table_name)):
            rospy.loginfo("{} file is not found in current present working directory".format(q_table_name))
            successfully_loaded = False
        else:
            with open(q_table_name, "rb") as q_table_file:
                self.q_table = pickle.load(q_table_file)
                successfully_loaded = True
        return successfully_loaded
            
    def get_state_tuple_from_dictionary(self, state_dictionary, directional_states):
        state = []
        for direction in directional_states:
            state.append(state_dictionary[direction])
        return tuple(state) 

    def get_policy(self, state, verbose = True, epsilon = 1.0):
        """ 
        Receives a Q_Table (as a dictionary) and a state and return the action
        dictated by the policy.

        Args:
            q_table: Q Table in form of a dictionary
            state_dictionary: current discretized state of the robot, in form of a dictionary
            verbose (bool, optional): Whether or not to notify users of the returned policy . Defaults to True.

        Returns:
            A tuple in format (angular speed, boolean indicating whether to reverse the robot or not) 
        """
        if state is not None and self.q_table is not None:
            # since the Q_Table takes in a tuple as the key
            # we must convert the state dictionary into a tuple
            state = self.get_state_tuple_from_dictionary(state, ["Front", "Upper Left", "Upper Right", "Lower Left", "Lower Right","Opponent Position"])

            # get q-values associated with different actions
            # and boolean flag indicating whether to reverse or not    
            q_values = self.q_table[tuple(state)]
            exploration_flag = False
            # generate an random number r 
            r = np.random.random()

            chosen_action = None
            # if r < epsilon, choose an action with highest q-value (utility)  
            actions = list(q_values.keys())
            if (r < epsilon):
                max_q_values = float("-inf")
                for action in actions :
                    if q_values[action] > max_q_values:
                        chosen_action = action
                        max_q_values = q_values[action]
            else:
                chosen_action = actions[int(np.random.random() * len(actions))]
                exploration_flag = True 
            
           
            translation_velocity, angular_velocity = chosen_action
           
            if verbose:
                if exploration_flag:
                    rospy.loginfo("Exploration. Random action chosen.")
                else:
                    rospy.loginfo("Exploitation. Choose action with max-utility.")
                rospy.loginfo("Action: translation: {}, angular {})".format(translation_velocity, angular_velocity))
            return translation_velocity, angular_velocity
        
        rospy.loginfo("Not returning valid action")
        return -1,-1

    
    def follow_policy(self, time_to_apply_action = 0.33):

        current_state = self.get_agent_state_function(self.agent_type)
        translation_velocity, angular_velocity = self.get_policy(current_state, verbose= False, epsilon = 1.0)
        self.agent_take_action_function(self.agent_type, translation_velocity, angular_velocity, time_to_apply_action)

    def learn(self, epsilon, time_to_apply_action = 0.33):
        # Learn using Q-Learning
        # does one q-value update
        current_state = self.get_agent_state_function(self.agent_type)
            
        rospy.loginfo("Epsilon: {}".format(epsilon))
        
        # get action A from S using policy
        chosen_action = tuple(self.get_policy(current_state, verbose = True, epsilon= epsilon))
        translation_speed, turn_action = chosen_action
        
        # take action A and move player, this would change the player's state
        self.agent_take_action_function(self.agent_type, translation_speed, turn_action, time_to_apply_action)
        
        # robot is now in new state S' 
        new_state = self.get_agent_state_function(self.agent_type)
        # robot now observes reward R(S') at this new state S'
        reward = self.reward_function(new_state)
        
        # update Q-value for Q(S,A)
        # Q(S,A) = Q(S,A) +  learning_rate*(reward + discount_factor* (argmax_A' Q(S', A')) - Q(S,A))
        directional_states= self.get_game_information("directional_states")

        current_state_tuple = self.get_state_tuple_from_dictionary(current_state,directional_states)
        new_state_tuple = self.get_state_tuple_from_dictionary(new_state,directional_states)

        best_action = tuple(self.get_policy(new_state, verbose=False, epsilon = 1.0))
        self.q_table[current_state_tuple][chosen_action] += self.learning_rate*(reward + (self.discount_factor * self.q_table[new_state_tuple][best_action]) - self.q_table[current_state_tuple][chosen_action])    
        
        return reward, current_state, translation_speed, turn_action
    
    def reward_function(self, state, verbose = True):
        # The reward function is the link between agent's learning algorithm and the environment so i can't completely separate them
        # To apply this Q-Learner to more problem, please write your own reward function here

        # get in-game information
        pursuer_radius = self.get_game_information("pursuer_radius")
        evader_radius = self.get_game_information("evader_radius")
        safe_distance_from_obstacle = self.get_game_information("safe_distance_from_obstacle")
        pursuer_stuck = self.get_game_information("pursuer_stuck")
        evader_stuck = self.get_game_information("evader_stuck")
        pursuer_was_stuck_but_rescued = self.get_game_information("pursuer_was_stuck_but_rescued")
        evader_was_stuck_but_rescued = self.get_game_information("evader_was_stuck_but_rescued")
        pursuer_min_distance_to_obstacle = self.get_game_information("pursuer_min_distance_to_obstacle")
        evader_min_distance_to_obstacle = self.get_game_information("evader_min_distance_to_obstacle")
        pursuer_min_distance_to_obstacle_direction = self.get_game_information("pursuer_min_distance_to_obstacle_direction")
        game_timeout = self.get_game_information("game_timeout")
        num_times_evader_stuck_in_episode = self.get_game_information("num_times_evader_stuck_in_episode")
        distance_between_players = self.get_game_information("distance_between_players")

        if self.agent_type == "pursuer":
            # if the pursuer gets stuck, it loses that game -> negative reward
            # the negative reward is also based on how badly it lost that round

            # the distance is between player calculated from the positions is the distance from one's robot's center to another robot's center
            # while distance gathered from the pursuer's LIDAR is from the pursuer's center to the evader's nearest SIDE 
            # thus, we need to adjust this DISTANCE_BETWEEN_PLAYERS by the evader's radius to better compare the two 
            true_distance_between_player = (distance_between_players - evader_radius)
            true_safe_distance_from_obstacle = (pursuer_radius + safe_distance_from_obstacle)

            # rospy.loginfo("PURSUER_MIN_DISTANCE: {}\nTRUE_DISTANCE_BETWEEN_PLAYERS: {}\nTRUE_SAFE_DISTANCE_FROM_OBSTACLE: {}".format(PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_DISTANCE_BETWEEN_PLAYERS ,TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
            if pursuer_stuck or pursuer_was_stuck_but_rescued:
                # rospy.loginfo("STUCK!")
                state_description = "STUCK!"
                reward = -30 * sigmoid(true_distance_between_player)
            elif state["Opponent Position"] == "Tagged":
                # rospy.loginfo("TAGGED!")
                state_description = "TAGGED!"
                reward = 30 

            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] in ["Close", "OK"] and state["Opponent Position"] in ["Front", "Close Front"]:
                
                state_description = "Obstacle on both sides, but there is opening in front and the target is in front nearby"
                reward = -sigmoid(1/pursuer_min_distance_to_obstacle) + 2*sigmoid(1/true_distance_between_player)
            
            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] == "Far" and state["Opponent Position"] == "Front":
                
                state_description = "Obstacle on both sides, but there is opening in front and the target is in front far away"
                reward = -sigmoid(1/pursuer_min_distance_to_obstacle) + sigmoid(1/true_distance_between_player) 
            
            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] != "Close":

                state_description = "Obstacle on both sides, but there is opening in front but opponent is not in front"
                reward = -sigmoid(1/pursuer_min_distance_to_obstacle) - sigmoid(true_distance_between_player) 
            
            # if there are obstacles nearby ON ONE SIDE(that is not the evader), and the evader is far away, promote obstacles avoidance behavior
            elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left")  or \
                    (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right") or \
                    (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left") or \
                    (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right"))
                ) and true_distance_between_player > true_safe_distance_from_obstacle\
                    and pursuer_min_distance_to_obstacle_direction != "Front": 

                state_description = "Obstacle is a lot nearer on the sides compared to evader. Prioritize obstacle avoidance"
                # extra punishment depending on how far the evader is and how close the pursuer is to an obstacle
                reward = -0.5 - sigmoid(true_distance_between_player) - sigmoid(1/pursuer_min_distance_to_obstacle)
            
            # there is an obstacle in front that is not the opponent
            elif (state["Front"] == "Close" or (pursuer_min_distance_to_obstacle_direction == "Front" and pursuer_min_distance_to_obstacle <= true_safe_distance_from_obstacle))\
                and state["Opponent Position"] not in  ["Front", "Close Front"]:
                
                state_description = "Obstacle directly infront that is not the opponent. Prioritize obstacle avoidance"
                reward = -0.25 - sigmoid(true_distance_between_player) - sigmoid(1/pursuer_min_distance_to_obstacle)
            
            # check for special case where opponent is directly in front, yet behind an obstacle, so robot priotize obstacle avoidance
            elif state["Front"] == "Close" and state["Opponent Position"] == "Front" \
                and true_distance_between_player >= pursuer_min_distance_to_obstacle\
                and pursuer_min_distance_to_obstacle < true_safe_distance_from_obstacle\
                and pursuer_min_distance_to_obstacle_direction == "Front":

                state_description = "Evader is directly in front, but probably is behind an obstacle. Prioritize obstacle avoidance"
                reward = -0.25 - sigmoid(true_distance_between_player) - sigmoid(1/pursuer_min_distance_to_obstacle)
            
            # else if the evader is in front and closeby, and we are relatively safe from obstacles on either sides
            elif state["Opponent Position"] == "Front" and true_distance_between_player <= 1.0 and\
                true_distance_between_player <= pursuer_min_distance_to_obstacle:
                # encourage robot to orient itself such that the opponent is directly in front of it
                # take away the sigmoid of the distance to encourage it to minimize such distance 
                state_description = "Evader is in front and close enough by, and we are relatively safe from obstacle!"
                reward = sigmoid(1.0/true_distance_between_player)
            
            elif state["Opponent Position"] == "Front":
                state_description = "Evader is in front but not that close"
                reward = 0.5* sigmoid(1/true_distance_between_player)
            
            # if the other robot is nearby and there is an obstacle, there is a chance that obstacle 
            # may be the other robot, so we encourage those states
            # or if the distance between players are very close
            elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left")  or \
                    (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
                    (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left") or \
                    (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
                    (state["Front"] in ["Close"] and state["Opponent Position"] == "Close Front")) and\
                        true_distance_between_player <= pursuer_min_distance_to_obstacle) or\
                            (true_distance_between_player <= true_safe_distance_from_obstacle):
                state_description = "Evader is nearby and we are relatively safe from obstacles"
                reward = 2.5 * sigmoid(1/true_distance_between_player)

            # there is no obstacle nearby and the target evader is far away
            elif state["Upper Left"] not in ["Close", "Too Close"] and state["Lower Left"] not in ["Close", "Too Close"]\
                and state["Upper Right"] not in ["Close", "Too Close"] and state["Lower Right"] not in ["Close", "Too Close"]\
                and state["Front"] != "Close" and true_distance_between_player >= safe_distance_from_obstacle:
                
                state_description = "Safe from obstacle but opponent is not nearby"
                reward = sigmoid(pursuer_min_distance_to_obstacle) - sigmoid(true_distance_between_player)  
            else:
                state_description = "Neutral state"
                reward = 0

        # REWARD FUNCTION FOR EVADER -----------------------------------------------------------------------------------------------------------------------------------
        elif self.agent_type == "evader":

            true_distance_between_player = (distance_between_players - pursuer_radius)
            true_safe_distance_from_obstacle = (evader_radius + safe_distance_from_obstacle)

            if evader_stuck or evader_was_stuck_but_rescued:
                state_description = "STUCK!"
                reward = -30 
            elif state["Opponent Position"] == "Tagged":
                state_description = "TAGGED!"
                reward = -30 
            elif game_timeout:
                state_description = "Game Timeout"
                reward = (30 * sigmoid(true_distance_between_player))/(num_times_evader_stuck_in_episode + 1)
            # avoid obstacle on all sides
            elif (state["Front"] == "Close"):
                state_description = "Obstacle really close by in front! Prioritize obstacle avoidance"
                reward = -0.5 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif (state["Upper Left"] == "Too Close") or\
                (state["Upper Right"] == "Too Close") or\
                (state["Lower Left"] == "Too Close") or\
                (state["Lower Right"] == "Too Close"):
                state_description = "Obstacle really closeby on the side. Prioritize obstacle avoidance"
                # extra punishments depending on how close the pursuer is and how close the evader is to an obstacle
                reward = -0.5 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            # encourage keeping a safe distance from obstacle
            elif ((state["Upper Left"] == "OK" and state["Lower Left"] == "OK") or (state["Upper Right"] == "OK" and state["Lower Right"] == "OK"))\
                and true_distance_between_player >= 0.75:
                state_description = "Robot is maintaing safe distance from obstacle on either side"
                reward = 0.5 + sigmoid(evader_min_distance_to_obstacle) - sigmoid(1/true_distance_between_player)
            elif state["Opponent Position"] == "Front" and true_distance_between_player <= 1.0:
                # discourage evader from moving toward the pursuer when they are within 1.0 unit from each other
                state_description = "Pursuer is in front within 1.0 unit of distance! Go the opposite direction"
                reward = -0.25 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif state["Opponent Position"] == "Bottom" and true_distance_between_player >= 1.0:
                state_description = "Pursue is behind and far away"
                reward = 0.5 + sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif state["Opponent Position"] == "Bottom" and true_distance_between_player >= 0.75:
                state_description = "Pursue is behind and decent distance away"
                reward = 0.25 + sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif "Close" not in state["Opponent Position"] and true_distance_between_player >= 0.75:
                state_description = "Pursuer is not close by we are relatively safe from obstacle"
                reward = 0.5 + sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif "Close" not in state["Opponent Position"]:
                state_description = "Pursuer is in vincinity but we are relatively safe from obstacle"
                reward = sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif "Close" in state["Opponent Position"]:
                state_description = "Pursuer is extremely close. Run away!"
                reward = -1 - 2 *sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            else:
                state_description = "Neutral state"
                reward = 0.25


        if verbose:
            # rospy.loginfo("DISTANCE BTW PLAYER: {}, PURSUER_MIN_DIST_OBSTACLE = {}, TRUE_SAFE_DISTANCE_FROM_OBSTACLE = {}".format(TRUE_DISTANCE_BETWEEN_PLAYERS, PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
            rospy.loginfo("{}'s state is {}".format(self.agent_type, state))
            rospy.loginfo("{}'s state's is {}".format(self.agent_type, state_description))
            rospy.loginfo("{}'s reward is {}".format(self.agent_type, reward))

        return reward 


    def save_agent(self, q_table_name):
        with open("{}".format(q_table_name), "w") as q_table_file:
            q_table_file.seek(0) # evader_rescue_thread.join()
            q_table_file.write(pickle.dumps(self.q_table))
        