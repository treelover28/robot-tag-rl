import os
import rospy
import cPickle as pickle
import numpy as np 
import tf
from mischallenous_functions import _get_permutations
from activation_functions import sigmoid
from base_agent import Base_Agent

# from robot_tag import random_walk_behavior

class Simple_Q_Learning_Agent(Base_Agent):
    def __init__(self, agent_type, learning_rate, discount_factor, substates_ratings_list, actions_list, get_agent_lidar_readings, agent_take_action_function, get_game_information):
        # call Agent's constructor
        super(Simple_Q_Learning_Agent, self).__init__(agent_type, learning_rate, discount_factor, get_agent_lidar_readings, agent_take_action_function, get_game_information)

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

    # @classmethod
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

    def get_current_state_discrete(self, verbose = False):
        lidar_readings = None 
        while(lidar_readings == None):
            # get lidar range-readings from game environment
            lidar_readings, _ = self.get_agent_lidar_readings(self.agent_type)
        
        
        # divide readings into 5 angular sector
        front_sector = range(0,30) + range(330,360)
        upper_left_sector = range(30,80)
        lower_left_sector = range(80,130)
        upper_right_sector = range(270,330)
        lower_right_sector = range(220,270) 
    
        # use the smallest distance detected at each directional state
        min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right = [float("inf") for i in range(0,5)]

        for angle in front_sector:
            min_front = min(min_front, lidar_readings[angle])
        for angle in upper_left_sector:
            min_upper_left = min(min_upper_left, lidar_readings[angle])
        for angle in lower_left_sector:
            min_lower_left = min(min_lower_left, lidar_readings[angle])
        for angle in upper_right_sector:
            min_upper_right = min(min_upper_right, lidar_readings[angle])
        for angle in lower_right_sector:
            min_lower_right = min(min_lower_right, lidar_readings[angle])
        

        # get pursuer and evader location from game environment
        # get pursuer and evader location from game environment
        pursuer_position, evader_position  = None, None
        while(pursuer_position == None or evader_position == None):
            pursuer_position = self.get_game_information("pursuer_position")
            evader_position = self.get_game_information("evader_position")

        if self.agent_type == "pursuer":    
            opp_rating, _, _ = self.get_opponent_position_rating(pursuer_position, evader_position)
        else:
            opp_rating, _, _ = self.get_opponent_position_rating(evader_position, pursuer_position)

        self.current_state_discrete = {
            "Front": self.get_distance_rating("Front", min_front, self.agent_type), \
            "Upper Left" : self.get_distance_rating("Upper Left", min_upper_left, self.agent_type), \
            "Upper Right": self.get_distance_rating("Upper Right", min_upper_right, self.agent_type), \
            "Lower Left": self.get_distance_rating( "Lower Left", min_lower_left, self.agent_type), \
            "Lower Right": self.get_distance_rating("Lower Right", min_lower_right, self.agent_type), \
            "Opponent Position": opp_rating
        }
     
        all_direction = ["Front", "Upper Left", "Lower Left", "Upper Right", "Lower Right"]
        all_distances =  [min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ]
        
        self.min_distance_to_obstacle = min(all_distances)
        index_ = all_distances.index(self.min_distance_to_obstacle)
        self.min_distance_to_obstacle_direction = all_direction[index_]

        if verbose:
            rospy.loginfo("Pursuer's state: {}".format(self.current_state_discrete))
            
        return self.current_state_discrete
        

    def get_opponent_position_rating(self, player_A, player_B):

        # get pursuer and evader location from game environment
        pursuer_position = self.get_game_information("pursuer_position")
        evader_position = self.get_game_information("evader_position")

        pursuer_radius = self.get_game_information("pursuer_radius")
        evader_radius = self.get_game_information("evader_radius")
        safe_distance_from_obstacle = self.get_game_information("safe_distance_from_obstacle")

        if player_A == pursuer_position:
            true_safe_distance_from_obstacle = pursuer_radius + safe_distance_from_obstacle
        else:
            true_safe_distance_from_obstacle = evader_radius + safe_distance_from_obstacle
        
        
        player_A_position = np.array(player_A[:3])
        player_A_orientation = np.array(player_A[3:])
        
        player_B_position = np.array(player_B[:3])
        player_B_orientation = np.array(player_B[3:])
        
        # the normal point upward in Z-direction
        plane_normal = np.array([0,0,1])

        # get current player's yaw
        _ ,_ , player_A_yaw = tf.transformations.euler_from_quaternion(player_A_orientation)
        
        # unit vector (from current player's coordinate frame) pointing in the direction current player orientation
        vector_player_pose = np.array([np.cos(player_A_yaw), np.sin(player_A_yaw), 0])
        # vector from current player to B
        vector_player_to_opponent = player_B_position - player_A_position
        # dot product between current player's pose vector and vector from current player to opponent
        dot_product = np.dot(vector_player_pose, vector_player_to_opponent)
        # length of the two vectors
        norm_player_pose = np.linalg.norm(vector_player_pose)
        norm_player_to_opponent = np.linalg.norm(vector_player_to_opponent)
        
        # angle between u and v = cos^{-1}(u . v / (|u||v|))
        # this is the angle between current angle and current opponent
        # this angle in degree is in [0, 180] regardless if it robot B to is to the left or right of robot A
        
        #
        #             |       |
        #             |       |
        # same angle( A       A ) same angle
        #           /           \ 
        #          /             \
        #       B                  B

        # but what we really want is this
        #
        #           |       |
        #           |       |
        #           A )      A ) angle
        #          /  reflex  \ 
        #         /   angle    \
        #       B                B

        angle_radian = np.arccos(dot_product/(np.dot(norm_player_pose,norm_player_to_opponent)))
        
        # surface normal of this "triangular" area 
        surface_normal = np.cross(vector_player_pose, vector_player_to_opponent)
        # to get the proper angle from the right-hand-side over
        # see second picture
        # we observe that when the angle between the two vectors are > 180, 
        # the surface normal of the triangle would be pointing downward, instead of in same direction as Z-axis
        # thus, we can dot the surface normal and the plane normal to check if they are in the same direction
        # if they are not, the dot product is negative -> we can go ahead and reverse this angle 
        
        if np.dot(plane_normal, surface_normal) < 0:
            angle_radian *= -1
        
        # now this angle in degree is in range[-180...180]
        angle_degree = np.rad2deg(angle_radian)
        
        # we make it to be in range by [0, 360] for easier intepretation
        # offsetting any negative angle by 360 
        if angle_degree < 0:
            angle_degree += 360 
        
        if 0 <= angle_degree < 30 or 330 <= angle_degree < 360:
            direction_rating = "Front"
        elif 30 <= angle_degree < 135:
            direction_rating = "Left"
        elif 225 <= angle_degree < 330:
            direction_rating = "Right"
        else:
            direction_rating = "Bottom"

        # get distance between player
        global DISTANCE_BETWEEN_PLAYERS
        distance_between_players =  np.linalg.norm(vector_player_to_opponent)
        
        if distance_between_players <= 0.3:
            rating = "Tagged"
        elif distance_between_players <= true_safe_distance_from_obstacle * 1.2:
            rating = "Close " + direction_rating
        else:
            rating = direction_rating

        return rating, distance_between_players, angle_degree

    
    def get_distance_rating(self, direction, distance, player_type):
        
        pursuer_radius = self.get_game_information("pursuer_radius")
        evader_radius = self.get_game_information("evader_radius")
        safe_distance_from_obstacle = self.get_game_information("safe_distance_from_obstacle")

        if player_type == "pursuer":
            true_safe_distance_from_obstacle = pursuer_radius + safe_distance_from_obstacle
        else:
            true_safe_distance_from_obstacle = evader_radius + safe_distance_from_obstacle

        if direction == "Front":
            interval = true_safe_distance_from_obstacle / 1.5
            if distance <= interval * 1.45:
                rating = "Close"
            elif distance <= interval * 2.2:
                rating = "OK"
            else:
                rating = "Far"
        elif direction in ["Left", "Right", "Upper Left", "Upper Right", "Lower Left", "Lower Right"]:
            interval = true_safe_distance_from_obstacle/ 2.5
            if distance <= interval * 1.6:
                rating = "Too Close"
            elif distance <= (interval * 2.25):
                rating = "Close"
            elif distance <= (interval * 3.67):
                rating = "OK"
            else:
                rating = "Far"
        
        return rating

    def follow_policy(self, time_to_apply_action = 0.33, verbose = False):
        current_state = self.get_current_state_discrete()
        
        if verbose:
            rospy.loginfo("Q-Learning agent's current state is {}".format(current_state))

        translation_velocity, angular_velocity = self.get_policy(current_state, verbose= False, epsilon = 1.0)
        self.agent_take_action_function(self.agent_type, translation_velocity, angular_velocity, time_to_apply_action)

    def learn(self, epsilon, time_to_apply_action = 0.33):
        # Learn using Q-Learning
        # does one q-value update
        
        current_state = self.get_current_state_discrete()
            
        rospy.loginfo("Epsilon: {}".format(epsilon))
        
        # get action A from S using policy
        chosen_action = tuple(self.get_policy(current_state, verbose = True, epsilon= epsilon))
        translation_speed, turn_action = chosen_action
        
        # take action A and move player, this would change the player's state
        self.agent_take_action_function(self.agent_type, translation_speed, turn_action, time_to_apply_action)
        
        # robot is now in new state S' 
        new_state = self.get_current_state_discrete()
        # robot now observes reward R(S') at this new state S'
        reward, _ = self.reward_function(new_state)
        
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
        game_timeout = self.get_game_information("game_timeout")
        num_times_evader_stuck_in_episode = self.get_game_information("num_times_evader_stuck_in_episode")
        distance_between_players = self.get_game_information("distance_between_players")

        is_terminal_state = False
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
                is_terminal_state = True
            elif state["Opponent Position"] == "Tagged":
                # rospy.loginfo("TAGGED!")
                state_description = "TAGGED!"
                reward = 30 
                is_terminal_state = True

            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] in ["Close", "OK"] and state["Opponent Position"] in ["Front", "Close Front"]:
                
                state_description = "Obstacle on both sides, but there is opening in front and the target is in front nearby"
                reward = -sigmoid(1/self.min_distance_to_obstacle) + 2*sigmoid(1/true_distance_between_player)

            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] == "Far" and state["Opponent Position"] == "Front":
                
                state_description = "Obstacle on both sides, but there is opening in front and the target is in front far away"
                reward = -sigmoid(1/self.min_distance_to_obstacle) + sigmoid(1/true_distance_between_player) 
            
            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] != "Close":

                state_description = "Obstacle on both sides, but there is opening in front but opponent is not in front"
                reward = -sigmoid(1/self.min_distance_to_obstacle) - sigmoid(true_distance_between_player) 
            
            # if there are obstacles nearby ON ONE SIDE(that is not the evader), and the evader is far away, promote obstacles avoidance behavior
            elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left")  or \
                    (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right") or \
                    (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left") or \
                    (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right"))
                ) and true_distance_between_player > true_safe_distance_from_obstacle\
                    and self.min_distance_to_obstacle_direction != "Front": 

                state_description = "Obstacle is a lot nearer on the sides compared to evader. Prioritize obstacle avoidance"
                # extra punishment depending on how far the evader is and how close the pursuer is to an obstacle
                reward = -0.5 - sigmoid(true_distance_between_player) - sigmoid(1/self.min_distance_to_obstacle)
            
            # there is an obstacle in front that is not the opponent
            elif (state["Front"] == "Close" or (self.min_distance_to_obstacle_direction == "Front" and self.min_distance_to_obstacle <= true_safe_distance_from_obstacle))\
                and state["Opponent Position"] not in  ["Front", "Close Front"]:
                
                state_description = "Obstacle directly infront that is not the opponent. Prioritize obstacle avoidance"
                reward = -0.25 - sigmoid(true_distance_between_player) - sigmoid(1/self.min_distance_to_obstacle)
            
            # check for special case where opponent is directly in front, yet behind an obstacle, so robot priotize obstacle avoidance
            elif state["Front"] == "Close" and state["Opponent Position"] == "Front" \
                and true_distance_between_player >= self.min_distance_to_obstacle\
                and self.min_distance_to_obstacle < true_safe_distance_from_obstacle\
                and self.min_distance_to_obstacle_direction == "Front":

                state_description = "Evader is directly in front, but probably is behind an obstacle. Prioritize obstacle avoidance"
                reward = -0.25 - sigmoid(true_distance_between_player) - sigmoid(1/self.min_distance_to_obstacle)
            
            # else if the evader is in front and closeby, and we are relatively safe from obstacles on either sides
            elif state["Opponent Position"] == "Front" and true_distance_between_player <= 1.0 and\
                true_distance_between_player <= self.min_distance_to_obstacle:
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
                        true_distance_between_player <= self.min_distance_to_obstacle) or\
                            (true_distance_between_player <= true_safe_distance_from_obstacle):
                state_description = "Evader is nearby and we are relatively safe from obstacles"
                reward = 2.5 * sigmoid(1/true_distance_between_player)

            # there is no obstacle nearby and the target evader is far away
            elif state["Upper Left"] not in ["Close", "Too Close"] and state["Lower Left"] not in ["Close", "Too Close"]\
                and state["Upper Right"] not in ["Close", "Too Close"] and state["Lower Right"] not in ["Close", "Too Close"]\
                and state["Front"] != "Close" and true_distance_between_player >= safe_distance_from_obstacle:
                
                state_description = "Safe from obstacle but opponent is not nearby"
                reward = sigmoid(self.min_distance_to_obstacle) - sigmoid(true_distance_between_player)  
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
                is_terminal_state = True 
            elif state["Opponent Position"] == "Tagged":
                state_description = "TAGGED!"
                reward = -30
                is_terminal_state = True
            elif game_timeout:
                state_description = "Game Timeout"
                reward = (30 * sigmoid(true_distance_between_player))/(num_times_evader_stuck_in_episode + 1)
            # avoid obstacle on all sides
            elif (state["Front"] == "Close"):
                state_description = "Obstacle really close by in front! Prioritize obstacle avoidance"
                reward = -0.75 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            elif (state["Upper Left"] == "Too Close") or\
                (state["Upper Right"] == "Too Close") or\
                (state["Lower Left"] == "Too Close") or\
                (state["Lower Right"] == "Too Close"):
                state_description = "Obstacle really closeby on the side. Prioritize obstacle avoidance"
                # extra punishments depending on how close the pursuer is and how close the evader is to an obstacle
                reward = -0.5 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            # encourage keeping a safe distance from obstacle
            elif ((state["Upper Left"] == "OK" and state["Lower Left"] == "OK") or (state["Upper Right"] == "OK" and state["Lower Right"] == "OK"))\
                and true_distance_between_player >= 0.75:
                state_description = "Robot is maintaing safe distance from obstacle on either side"
                reward = 0.5 + sigmoid(self.min_distance_to_obstacle) - sigmoid(1/true_distance_between_player)
            elif state["Opponent Position"] == "Front" and true_distance_between_player <= 1.0:
                # discourage evader from moving toward the pursuer when they are within 1.0 unit from each other
                state_description = "Pursuer is in front within 1.0 unit of distance! Go the opposite direction"
                reward = -0.25 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            elif state["Opponent Position"] == "Bottom" and true_distance_between_player >= 1.0:
                state_description = "Pursue is behind and far away"
                reward = 0.5 + sigmoid(true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            elif state["Opponent Position"] == "Bottom" and true_distance_between_player >= 0.75:
                state_description = "Pursue is behind and decent distance away"
                reward = 0.25 + sigmoid(true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            elif "Close" not in state["Opponent Position"] and true_distance_between_player >= 0.75:
                state_description = "Pursuer is not close by we are relatively safe from obstacle"
                reward = 0.5 + sigmoid(true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            elif "Close" not in state["Opponent Position"]:
                state_description = "Pursuer is in vincinity but we are relatively safe from obstacle"
                reward = sigmoid(true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            elif "Close" in state["Opponent Position"]:
                state_description = "Pursuer is extremely close. Run away!"
                reward = -1 - 2 *sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/self.min_distance_to_obstacle)
            else:
                state_description = "Neutral state"
                reward = 0.25


        if verbose:
            # rospy.loginfo("DISTANCE BTW PLAYER: {}, PURSUER_MIN_DIST_OBSTACLE = {}, TRUE_SAFE_DISTANCE_FROM_OBSTACLE = {}".format(TRUE_DISTANCE_BETWEEN_PLAYERS, PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
            rospy.loginfo("{}'s state is {}".format(self.agent_type, state))
            rospy.loginfo("{}'s state's is {}".format(self.agent_type, state_description))
            rospy.loginfo("{}'s reward is {}".format(self.agent_type, reward))

        return (reward, is_terminal_state)


    def save_agent(self, q_table_name):
        with open("{}".format(q_table_name), "w") as q_table_file:
            q_table_file.seek(0) # evader_rescue_thread.join()
            q_table_file.write(pickle.dumps(self.q_table))
        