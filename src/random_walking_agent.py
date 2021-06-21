from base_agent import Base_Agent
import rospy
import numpy as np
import tf


class Random_Walking_Agent(Base_Agent):
    def __init__(self, agent_type, get_agent_lidar_readings, agent_take_action_function, get_game_information, random_action_chance):
        super(Random_Walking_Agent, self).__init__(agent_type, None, None, get_agent_lidar_readings, agent_take_action_function, get_game_information)
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
        robot_state = self.get_current_state_discrete()

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
            rospy.loginfo("Pursuer's state: {}".format(self.current_state))
            
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
