#! /usr/bin/python

import rospy
import os.path
import cPickle as pickle
import tf
import sys
import threading 

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry


import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from math import pi



PURSUER_STATE_DISCRETIZED = None 
PURSUER_STATE_CONTINUOUS = None
PURSUER_POSITION = None 
PURSUER_STUCK = False

EVADER_STATE_DISCRETIZED = None 
EVADER_STATE_CONTINUOUS = None
EVADER_STUCK = False
EVADER_POSITION = None 

EVADER_DISTANCE_FROM_NEAREST_OBSTACLE = None
PURSUER_MIN_DISTANCE_TO_OBSTACLE = None 
DISTANCE_BETWEEN_PLAYERS = None

# for ros_plaza
# STARTING_LOCATIONS = [(0,1.2), (-2,1), (0,-1), (0,1.5), (0,-2), (-2,-1), (0.5,0), (-2,1.8),(1,0), (1,-2)]

# for ros pillars map
# STARTING_LOCATIONS = [(0,1), (-1,0), (0,-1), (1,0), (-1,-2), (-1,2)]

STARTING_LOCATIONS = [(0,0.5), (0.5,0), (0,-0.5), (0.5,0), (-1,-2), (-1,2)]

# State Space Hyperparameters
SAFE_DISTANCE_FROM_OBSTACLE = 0.3
ROTATIONAL_ACTIONS = [60,45,20,0,-20,-45,-60]
# slow speed 0.1 to help it slow down when near obstacle
# regular speed is 0.2 
# accelerated speed to help it speed up and catch the evader when it is nearby
TRANSLATION_SPEED = [0.075, 0.2, 0.35]
DIRECTIONAL_STATES = ["Front", "Upper Left", "Upper Right", "Lower Left", "Lower Right","Opponent Position"]
FRONT_RATINGS = ["Close", "OK", "Far"]
UPPER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
UPPER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
LOWER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
LOWER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
OPPONENT_RATINGS = ["Close Left", "Left", "Close Front", "Front", "Right", "Close Right", "Bottom", "Close Bottom", "Tagged"]

# DIFFERENT TOPICS SUBSCRIBERS AND LISTENERS
PURSUER_SCAN_SUBSCRIBER = None 
PURSUER_POSITION_SUBSCRIBER = None 
PURSUER_CMD_PUBLISHER = None 
EVADER_SCAN_SUBSCRIBER = None 
EVADER_POSITION_SUBSCRIBER = None 
EVADER_CMD_PUBLISHER = None 

# Q-tables
Q_TABLE_PURSUER = None 
Q_TABLE_EVADER = None 

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def reward_function(player_type, state, verbose = True):
    global DISTANCE_BETWEEN_PLAYERS
    DISTANCE_BETWEEN_PLAYERS = np.linalg.norm(np.array(PURSUER_POSITION[0:2]) - np.array(EVADER_POSITION[0:2]))
    
    
    if player_type == "pursuer":
        # if the pursuer gets stuck, it loses that game -> negative reward
        # the negative reward is also based on how badly it lost that round
        if PURSUER_STUCK:
            rospy.loginfo("STUCK!")
            state_description = "STUCK!"
            reward = -10 - 5*sigmoid(DISTANCE_BETWEEN_PLAYERS)
        elif state["Opponent Position"] == "Tagged":
            rospy.loginfo("TAGGED!")
            state_description = "TAGGED!"
            reward = 30 
        # if there are obstacles nearby (that is not the evader), and the evader is far away, promote obstacles avoidance behavior
        elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left")  or \
               (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right") or \
               (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left") or \
               (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right") or \
               (state["Front"] in ["Close"] and state["Opponent Position"] != "Close Front")) and\
                   DISTANCE_BETWEEN_PLAYERS > PURSUER_MIN_DISTANCE_TO_OBSTACLE * 1.25): 
            
            state_description = "Obstacle is a lot nearer compared to evader. Prioritize obstacle avoidance"
            # extra punishment depending on how far the evader is and how close the pursuer is to an obstacle
            reward = -1 - sigmoid(DISTANCE_BETWEEN_PLAYERS) - sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE)
            # if the other robot is nearby and there is an obstacle, there is a chance that obstacle 
            # may be the other robot, so we encourage those states
            # or if the distance between players are very close
        elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left")  or \
               (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
               (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left") or \
               (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
               (state["Front"] in ["Close"] and state["Opponent Position"] != "Close Front")) and\
                   DISTANCE_BETWEEN_PLAYERS <= PURSUER_MIN_DISTANCE_TO_OBSTACLE * 1.3) or\
                       (DISTANCE_BETWEEN_PLAYERS <= SAFE_DISTANCE_FROM_OBSTACLE * 1.2):
            state_description = "Evader is nearby and we are relatively safe from obstacles"
            reward = sigmoid(1/DISTANCE_BETWEEN_PLAYERS) * 2.5 
        elif state["Opponent Position"] == "Front" and DISTANCE_BETWEEN_PLAYERS <= 1.0:
            # encourage robot to orient itself such that the opponent is directly in front of it
            # take away the sigmoid of the distance to encourage it to minimize such distance 
            state_description = "Evader is in front and close enough by!"
            reward = sigmoid(1/DISTANCE_BETWEEN_PLAYERS)
        # there is no obstacle nearby and the target evader is far away
        elif DISTANCE_BETWEEN_PLAYERS >= PURSUER_MIN_DISTANCE_TO_OBSTACLE and PURSUER_MIN_DISTANCE_TO_OBSTACLE >= SAFE_DISTANCE_FROM_OBSTACLE:
            state_description = "No obstacle nearby and evader is also not nearby"
            reward = -0.5 - sigmoid(DISTANCE_BETWEEN_PLAYERS)
        else:
            state_description = "Neutral state"
            reward = 0
    elif player_type == "evader":
        # TODO: FIX THIS LATER
        # Evader loses the game if it gets stuck
        if EVADER_STUCK:
            reward = -2
        elif state["Opponent Position"] == "Tagged":
            reward = -5
        # to also promote obstacle avoidance, when it is far away from the pursuer
        elif (state["Left"] in ["Close", "Too Close"] or \
           state["Right"] in ["Close", "Too Close"] or \
           state["Front"] in ["Close"] or\
           EVADER_DISTANCE_FROM_NEAREST_OBSTACLE < SAFE_DISTANCE_FROM_OBSTACLE) \
               and DISTANCE_BETWEEN_PLAYERS > EVADER_DISTANCE_FROM_NEAREST_OBSTACLE:
           reward = -1
        # if there is no obstacles nearby and the evader is far away from the pursuer
        elif DISTANCE_BETWEEN_PLAYERS > PURSUER_MIN_DISTANCE_TO_OBSTACLE:
            reward = sigmoid(DISTANCE_BETWEEN_PLAYERS)
        # punish states where the tagee lets the pursuer gets too close 
        elif (DISTANCE_BETWEEN_PLAYERS <= SAFE_DISTANCE_FROM_OBSTACLE * 1.3) or \
             state["Opponent Position"] in ["Close Left", "Close Front", "Close Bottom", "Close Right"]:
            reward = -2.5 * sigmoid(1/DISTANCE_BETWEEN_PLAYERS)
        else:
            reward = 0 
    if verbose:
        rospy.loginfo("Distance between player: {} vs distance to obstacle: {}, safe distance from obstacle is {}".format(DISTANCE_BETWEEN_PLAYERS, PURSUER_MIN_DISTANCE_TO_OBSTACLE * 1.1, SAFE_DISTANCE_FROM_OBSTACLE))
        rospy.loginfo("{}'s state is {}".format(player_type, state_description))
        rospy.loginfo("{}'s reward is {}".format(player_type, reward))
    
    return reward 


def get_opponent_position_rating(player_A, player_B):
    
    player_A_position = np.array(player_A[:3])
    player_A_orientation = np.array(player_A[3:])
    
    player_B_position = np.array(player_B[:3])
    player_B_orientation = np.array(player_B[3:])
    
    plane_normal = np.array([0,0,1])

    _ ,_ , player_A_yaw = tf.transformations.euler_from_quaternion(player_A_orientation)
   
    vector_A = np.array([np.cos(player_A_yaw), np.sin(player_A_yaw), 0])
    vector_B = player_B_position - player_A_position
     
    dot_product = np.dot(vector_A, vector_B)
    
    norm_A = np.linalg.norm(vector_A)
    norm_B = np.linalg.norm(vector_B)
    
    angle_rad = np.arccos(dot_product/(np.dot(norm_A,norm_B)))
    cross = np.cross(vector_A, vector_B)
    
    if (np.dot(plane_normal, cross) < 0):
        angle_rad *= -1 
    
    angle_deg = np.rad2deg(angle_rad)
    if angle_deg < 0:
        angle_deg += 360 
    
    distance = np.linalg.norm(vector_B)

    if distance <= SAFE_DISTANCE_FROM_OBSTACLE:
        return "Tagged"

    # if player_A == PURSUER_POSITION:
    #     print("angle: {}".format(angle_deg))
    if 0 <= angle_deg < 30 or 330 <= angle_deg < 360:
       direction_rating = "Front"
    elif 30 <= angle_deg < 135:
        direction_rating = "Left"
    elif 225 <= angle_deg < 330:
        direction_rating = "Right"
    else:
        direction_rating = "Bottom"
    
    distance_rating = ""
    if distance <= SAFE_DISTANCE_FROM_OBSTACLE * 1.3:
        distance_rating = "Close"
    
    return (distance_rating + " " + direction_rating).strip()

def get_distance_rating(direction, distance):
    if direction == "Front":
        interval = SAFE_DISTANCE_FROM_OBSTACLE
        if distance <= (interval * 1.5):
            rating = "Close"
        else:
            rating = "Far"
    elif direction in ["Left", "Right", "Upper Left", "Upper Right", "Lower Left", "Lower Right"]:
        interval = SAFE_DISTANCE_FROM_OBSTACLE/2
        if distance <= interval * 1.5:
            rating = "Too Close"
        elif distance <= (interval * 2.2):
            rating = "Close"
        elif distance <= (interval * 3.5):
            rating = "OK"
        else:
            rating = "Far"
    return rating


def get_current_state(message,args):
    ranges_data = message.ranges

    player_type = args["player_type"]
    verbose = args["verbose"]

    front_sector = range(0,45) + range(315,360)
    
    upper_left_sector = range(45,90)
    lower_left_sector = range(90,135)
   
    upper_right_sector = range(270,315)
    lower_right_sector = range(225,270) 
   
    # use the smallest distance detected at each directional state
    min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right = [float("inf") for i in range(0,5)]

    for angle in front_sector:
        min_front = min(min_front, ranges_data[angle])
    for angle in upper_left_sector:
        min_upper_left = min(min_upper_left, ranges_data[angle])
    for angle in lower_left_sector:
        min_lower_left = min(min_lower_left, ranges_data[angle])
    for angle in upper_right_sector:
        min_upper_right = min(min_upper_right, ranges_data[angle])
    for angle in lower_right_sector:
        min_lower_right = min(min_lower_right, ranges_data[angle])
    

    if player_type == "pursuer":
        global PURSUER_STATE_DISCRETIZED 
        
        PURSUER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front), \
            "Upper Left" : get_distance_rating("Upper Left", min_upper_left), \
            "Upper Right": get_distance_rating("Upper Right", min_upper_right), \
            "Lower Left": get_distance_rating( "Lower Left", min_lower_left), \
            "Lower Right": get_distance_rating("Lower Right", min_lower_right), \
            "Opponent Position": get_opponent_position_rating(PURSUER_POSITION, EVADER_POSITION)
        }

        global PURSUER_MIN_DISTANCE_TO_OBSTACLE 
        PURSUER_MIN_DISTANCE_TO_OBSTACLE= min([ min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ])
        
        if verbose:
            rospy.loginfo("Pursuer's state: {}".format(PURSUER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of pursuer's state: {}".format(reward_function("pursuer", PURSUER_STATE_DISCRETIZED, verbose=True)))
    else:
        global EVADER_STATE_DISCRETIZED 
        EVADER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front), \
            "Upper Left" : get_distance_rating("Upper Left", min_upper_left), \
            "Upper Right": get_distance_rating("Upper Right", min_upper_right), \
            "Lower Left": get_distance_rating( "Lower Left", min_lower_left), \
            "Lower Right": get_distance_rating("Lower Right", min_lower_right), \
            "Opponent Position": get_opponent_position_rating(EVADER_POSITION, PURSUER_POSITION)
        }

        global EVADER_DISTANCE_FROM_NEAREST_OBSTACLE 
        EVADER_DISTANCE_FROM_NEAREST_OBSTACLE= min([ min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ])

        if verbose:
            rospy.loginfo("Evader's state: {}".format(EVADER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of evader's state: {}".format(reward_function("evader", EVADER_STATE_DISCRETIZED, verbose=False)))

def get_robot_location(message, args):
    player_type = args["player_type"]
    verbose = args["verbose"]
    
    position = [message.pose.pose.position.x, message.pose.pose.position.y, 0, \
         message.pose.pose.orientation.x, message.pose.pose.orientation.y,message.pose.pose.orientation.z, message.pose.pose.orientation.w]
    if player_type == "pursuer":
        global PURSUER_POSITION
        PURSUER_POSITION = position
        if verbose:
            rospy.loginfo("Pursuer's position is {}".format(PURSUER_POSITION))
    else:
        global EVADER_POSITION
        EVADER_POSITION = position
        if verbose:
            rospy.loginfo("Evader's position is {}".format(EVADER_POSITION))

def create_q_table(player_type = "pursuer"):
    """
    Generate a Q-Table in form of a dictionary where the key is a unique state in all O(m^n) possible states from resulting
    from n possible directions and each direction has m possible distance ratings. The value associated with each key (state)
    is a tuple consisting of (another tuple containing n q-values associated with n actions for each state, and a boolean to tell robot 
    whether to reverse or not). 

    The q-values here are manually defined (prior expert knowledge) where states are logically broken down into common scenarios a robot may face 
    while wall-following. Each common scenario has a desirable action that could address the scenario, and thus the associated q-values of such action
    is a lot higher than the q-values for other neutral actions.
 
    """
    # FRONT_RATINGS = ["Close", "OK", "Far"]
    # UPPER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
    # UPPER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
    # LOWER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
    # LOWER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
    
    q_table = {}
    all_states = []
    _get_permutations(0, [FRONT_RATINGS,UPPER_LEFT_RATINGS, UPPER_RIGHT_RATINGS, LOWER_LEFT_RATINGS, LOWER_RIGHT_RATINGS, OPPONENT_RATINGS], list(), 6, all_states)
    
    all_actions = []
    _get_permutations(0,[TRANSLATION_SPEED, ROTATIONAL_ACTIONS],list(),2, all_actions)

    for state in all_states:
        # unpack tuple to get corresponding distance ratings for each direction
        front_rating, upper_left_rating, upper_right_rating, lower_left_rating, lower_right_rating, opponent_rating = state
        state_dictionary = ({"Front": front_rating,  "Upper Left": upper_left_rating, \
                             "Upper Right": upper_right_rating, "Lower Left": lower_left_rating, \
                             "Lower Right": lower_right_rating, "Opponent Positon": opponent_rating})
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
    with open("q_table_{}.txt".format(player_type), "w") as q_table_file:
        q_table_file.seek(0)
        q_table_file.write(pickle.dumps(q_table)) 

def replace_speed_in_q_table(q_table_name, old_speed, new_speed):
    with open(q_table_name, "rb") as q_table_file:
        # loads copy of q-table in
        q_table = pickle.load(q_table_file)
        for state in q_table:
            action_q_values = q_table[state]
            for action in action_q_values:
                speed_0, angle_0 = action
                if speed_0 == old_speed:
                    # create new action as key
                    new_action = (new_speed, angle_0)
                    # remove old action and retrives its q-value, set to 0 if old action is not found => this scenerio will never happen
                    q_value = action_q_values.pop(action,0)
                    # re-insert new action as new key with the q_value
                    action_q_values[new_action] = q_value
    with open("replaced_{}".format(q_table_name), "w") as replacement_q_table_file:
        replacement_q_table_file.seek(0)
        replacement_q_table_file.write(pickle.dumps(q_table))
    





def _get_permutations(current_list_index, lists, prefix, k, states_accumulator):
    """ Helper function to generate all permutations of length n with replacements
    using elements in choices list. The prefix is the current partial permutation
    and k indicates how many more elements we still need to add to current partial permuation
    to finish it.

    Args:
        choices: list of elements to create permuation from
        prefix: contains the partial permuation
        k: (int) number of element left to add to current partial permutation to get it up to desireable length
        states_accumulator ([type]): list to accumulate / store the states.
    """
    if (k == 0):
        states_accumulator.append(prefix)
        return
    
    list_to_select_from = lists[current_list_index]
   
    for i in range(len(list_to_select_from)):
        new_prefix = (prefix + [list_to_select_from[i]])
        _get_permutations(current_list_index + 1, lists, new_prefix, k-1, states_accumulator)

def move_robot(player_type, translation_speed, angular_speed_degrees):
    """ Receive a linear speed and an angular speed (degrees/second), craft a Twist message,
    and send it to the /cmd_vel  topic to make the robot move
    """
    twist_message = Twist()
    twist_message.linear.x = translation_speed
    twist_message.linear.y = 0
    twist_message.linear.z = 0
    twist_message.angular.x = 0
    twist_message.angular.z = np.deg2rad(angular_speed_degrees)
    twist_message.angular.y = 0
    
    if (player_type == "pursuer" and PURSUER_CMD_PUBLISHER is not None):
        PURSUER_CMD_PUBLISHER.publish(twist_message)    
    elif (player_type == "evader" and EVADER_CMD_PUBLISHER is not None):   
        EVADER_CMD_PUBLISHER.publish(twist_message)


def set_robot_position(model_name, pose):
    position_x, position_y = pose 

    robot_position_msg = ModelState()
    robot_position_msg.model_name = model_name
    robot_position_msg.pose.position.x = position_x
    robot_position_msg.pose.position.y = position_y
    robot_position_msg.pose.orientation.w = 1.0
    robot_position_msg.reference_frame = "world"

    try: 
        set_pose = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        resp = set_pose(robot_position_msg)
    except rospy.ServiceException:
        rospy.loginfo("Service /gazebo/set_model_state failed")


def get_state_tuple_from_dictionary(state_dictionary):
    state = []
    for direction in DIRECTIONAL_STATES:
        state.append(state_dictionary[direction])
    return tuple(state) 

def get_policy(q_table, state_dictionary, verbose = True, epsilon = 1.0):
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
    if state_dictionary is not None and q_table is not None:
        # since the Q_Table takes in a tuple as the key
        # we must convert the state dictionary into a tuple
        state = get_state_tuple_from_dictionary(state_dictionary)

        # get q-values associated with different actions
        # and boolean flag indicating whether to reverse or not    
        q_values = q_table[tuple(state)]
        exploration_flag = False
        # generate an random number r 
        r = random.random()

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
            chosen_action = actions[int(random.random() * len(actions))]
            exploration_flag = True 
        
        # check if action returned is a tuple of (translation_velocity, angular_velocity) to be backward compatible with the older q-table
        # assuming that the state discretizations are not changed.
        # I fixed my translational speed to be 0.2 during my previous attempts, so the chosen_action back then was a single integer
        # indicating the angular velocity
        if isinstance(chosen_action, (tuple,list)):
            translation_velocity, angular_velocity = chosen_action
        else:
            translation_velocity, angular_velocity = 0.2, chosen_action
        
        if verbose:
            if exploration_flag:
                rospy.loginfo("Exploration. Random action chosen.")
            else:
                rospy.loginfo("Exploitation. Choose action with max-utility.")
            rospy.loginfo("Action: translation: {}, angular {})".format(translation_velocity, angular_velocity))
        return translation_velocity, angular_velocity
    
    rospy.loginfo("Not returning valid action")
    return -1,-1

def q_learning_td(player_type, q_table, learning_rate, epsilon, discount_factor, time_to_apply_action = 0.33):
    # does one training episode
    if player_type == "pursuer":
        current_state = PURSUER_STATE_DISCRETIZED 
        player_position = np.array(PURSUER_POSITION[:2])
        opponent_position = np.array(EVADER_POSITION[:2])
    else:
        current_state = EVADER_STATE_DISCRETIZED 
        player_position = np.array(EVADER_POSITION[:2])
        opponent_position = np.array(PURSUER_POSITION[:2])    
    
    rospy.loginfo("Epsilon: {}".format(epsilon))
    # rospy.loginfo("{}_current State: {}".format(player_type , current_state))
    # get action A from S using policy
    chosen_action = get_policy(q_table, current_state, verbose = False, epsilon= epsilon)
    translation_speed, turn_action = chosen_action
    # take action A and move player, this would change the player's state
    # rospy.loginfo("Chosen action {}".format(action))
    move_robot(player_type, translation_speed, turn_action)
    # give the robot some time to apply action => proper state transition
    rospy.sleep(time_to_apply_action)
    
    # robot is now in new state S' and observe reward R(S') 
    new_state = PURSUER_STATE_DISCRETIZED if (player_type == "pursuer") else EVADER_STATE_DISCRETIZED
    reward = reward_function(player_type, new_state)
    # rospy.loginfo("{}'s reward: {}".format(player_type, reward))
    
    # update Q-value for Q(S,A)
    # Q(S,A) = Q(S,A) +  learning_rate*(reward + discount_factor* (argmax_A' Q(S', A')) - Q(S,A))
    current_state_tuple = get_state_tuple_from_dictionary(current_state)
    new_state_tuple = get_state_tuple_from_dictionary(new_state)

    best_action = get_policy(q_table, new_state, verbose=False, epsilon = 1.0)
    q_table[current_state_tuple][chosen_action] += learning_rate*(reward + \
                    discount_factor * q_table[new_state_tuple][best_action] - q_table[current_state_tuple][chosen_action])    
    
    return reward


def random_walk_behavior(robot_type, robot_state, random_action_chance = 0.2):
    if robot_type == "evader":
        is_stuck = EVADER_STUCK
    else:
        is_stuck = PURSUER_STUCK

    if robot_state["Front"] == "Close" and robot_state["Upper Left"] == "Too Close" and robot_state["Upper Right"] == "Too Close" and is_stuck:
        translation_speed = -0.1 
        turn_angle = -60
    elif robot_state["Front"] == "Close" and is_stuck:
        translation_speed = -0.35
        turn_angle = -60
    else:
        translation_speed = 0.25 
        # 20% chance of making random turns
        if (random.random() < 0.2):
            turn_angle = random.randint(0,359)
        else:
            # just go straight else
            turn_angle = 0 
    # rospy.loginfo("translation_speed: {}, turn_angle {}".format(translation_speed, turn_angle))
    return (translation_speed, turn_angle)


def follow_policy(player_type, q_table, time_to_apply_action = 0.33):
    if player_type == "pursuer":
        current_state = PURSUER_STATE_DISCRETIZED
        translation_velocity, angular_velocity = get_policy(q_table, current_state, verbose= False, epsilon = 1.0)

    elif player_type == "evader":
        current_state = EVADER_STATE_DISCRETIZED
        translation_velocity, angular_velocity = random_walk_behavior(robot_type="evader", robot_state=current_state)
    
    move_robot(player_type, translation_velocity, angular_velocity)
    rospy.sleep(time_to_apply_action)

def is_stuck(last_few_positions, robot_state):
    # Checking if the robot is stuck requires info about 
    # whether it is near an obstacle and if its location has not changed in a while.
    #  
    # Checking if the location hasn't changed alone is not sufficient 
    # since the robot could be moving very slowly => the algorithm thinks it is stuck
    is_stuck = False
    if last_few_positions is not None:
        changes_in_x = 0
        changes_in_y = 0
        for i in range(1,len(last_few_positions)):
            changes_in_x += abs(last_few_positions[i][0] - last_few_positions[i - 1][0])
            changes_in_y += abs(last_few_positions[i][1] - last_few_positions[i - 1][1])
        # if accumulated changes in both coordinates are less than a very small number, 
        # the robot is probably stuck
        # is_near_obstacle = (robot_state["Upper Left"]  == "Too Close" or \
        #        robot_state["Upper Right"] == "Too Close" or \
        #        robot_state["Lower Left"] == "Too Close"  or \
        #        robot_state["Lower Right"] == "Too Close" or \
        #        robot_state["Front"] == "Close")
        
        is_near_obstacle = robot_state["Front"] == "Close" 
    
        is_in_same_place = changes_in_x < 0.05 and changes_in_y < 0.05
        # the robot is consider stuck of it is near an obstacle and hasn't changed position in a while
        is_stuck = is_near_obstacle and is_in_same_place
    return is_stuck

def is_terminal_state(train_type, time_elapsed, episode_time_limit, pursuer_stuck, evader_stuck, opponent_rating):
    if opponent_rating == "Tagged": 
        is_terminal = True
        rospy.loginfo("Terminated because TAGGED")
    elif time_elapsed >= rospy.Duration(secs = episode_time_limit):
        is_terminal = True
        rospy.loginfo("Terminated because {} minutes has passed".format(int(episode_time_limit/60)))
    # if we are just training the pursuer, even if the evader gets stuck
    # we still let the pursuer run until it catches the evader, or gets stucks itself
    elif train_type == "pursuer" and pursuer_stuck:
        is_terminal = True
        rospy.loginfo("Terminated because pursuer is STUCK")
    # # if we are training both, we end when either gets stuck
    # elif train_type in ["both", "evader"] and (pursuer_stuck or evader_stuck):
    #     is_terminal = True
    else:
        is_terminal = False 
    return is_terminal
    
def spawn_robots():
    # spawn pursuers and evaders at different locations throughout the map 
    pursuer_position = None 
    evader_position = None 
    while (pursuer_position == evader_position):
        pursuer_position = STARTING_LOCATIONS[random.randint(0, len(STARTING_LOCATIONS) - 1)]
        evader_position = STARTING_LOCATIONS[random.randint(0, len(STARTING_LOCATIONS) - 1)]
    
    set_robot_position("pursuer", pursuer_position)
    set_robot_position("evader", evader_position)

def _plot_learning_curve(line_chart, new_x, new_y):
    line_chart.set_xdata(np.append(line_chart.get_xdata(), new_x))
    line_chart.set_ydata(np.append(line_chart.get_ydata(), new_y))
    plt.draw()
    plt.pause(0.001)

def train(train_type = "both", total_episodes = 1000, learning_rate = 0.2, discount_factor = 0.8, starting_epsilon = 0.2, max_epsilon = 0.9, episode_time_limit = 30, time_to_apply_action=0.33):
    current_episode = 0
    accumulated_pursuer_reward = 0
    accumulated_evader_reward = 0
    epsilon = starting_epsilon

    # plot learning curve as robot learns
    learning_curve,  = plt.plot([],[], "r-", label="Q-learning TD")
    test_curve, = plt.plot([],[], linestyle="-", marker="x", color="k", label="Q-learning TD Test-Phase Reward")
    plt.xlabel("Training episode")
    plt.ylabel("Accumulated rewards")
    plt.xlim(0 , total_episodes)
    plt.ylim(-100, 100)
    plt.legend(loc="upper left")
    plt.axhline(y= 0, color = "g", linestyle = "-")
    plt.show(block=False)

    num_tagged = 0
    best_test_score = float("-inf")
    best_train_score = float("-inf")
    training_reward = 0
    while current_episode < total_episodes:
        if (PURSUER_STATE_DISCRETIZED is not None and EVADER_STATE_DISCRETIZED is not None):
            rospy.loginfo("Starting Episode {}".format(current_episode))
            print("*"*50)

            if train_type ==  "both":
                player_to_train = "pursuer" if (current_episode % 2 == 0) else "evader"
                opponent_to_test = "pursuer" if  player_to_train == "evader" else "evader"
                q_table_player = Q_TABLE_PURSUER if (player_to_train == "pursuer") else Q_TABLE_EVADER
                q_table_opponent = Q_TABLE_PURSUER if (player_to_train == "evader") else Q_TABLE_EVADER
            elif train_type == "pursuer":
                player_to_train = "pursuer"
                opponent_to_test = "evader"
                q_table_player = Q_TABLE_PURSUER
                q_table_opponent = Q_TABLE_EVADER
            else:
                player_to_train = "evader"
                opponent_to_test = "pursuer"
                q_table_player = Q_TABLE_EVADER
                q_table_opponent = Q_TABLE_PURSUER


            rospy.loginfo("Player being trained {}".format(player_to_train))
            # keep track of whether pursuer and evader are stuck, and what time
            global EVADER_STUCK
            global PURSUER_STUCK

            PURSUER_STUCK = False
            EVADER_STUCK = False

            last_few_pursuer_positions = []
            last_few_evader_positions = []

            # spawn robots at semi-random locations
            spawn_robots()

            # every 100 episodes, test the policy learned so far 
            # including episode 0 => baseline for how robot is doing pre-training
            if current_episode % 100 == 0:
                rospy.loginfo("Testing policy learned so far")
                test_reward = test(player_to_train, total_episodes = 10, episode_time_limit=episode_time_limit)
                if test_reward > best_test_score:
                    # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                    # save q-table
                    with open("q_table_{}_best_testing.txt".format(player_to_train), "w") as q_table_file:
                        q_table_file.seek(0)
                        q_table_file.write(pickle.dumps(q_table_player)) 
                    best_test_score = test_reward

                _plot_learning_curve(test_curve, current_episode, test_reward)
            
        
            start_time = rospy.Time.now()
            time_elapsed = rospy.Duration(secs=0)

            # every <epsilon_update_interval> training episodes, the epsilon goes up by 0.05 to encourage more exploitation and less exploration
            # as the robot learns more and more about the environment
            epsilon_update_interval = int(total_episodes / ((max_epsilon - starting_epsilon) / 0.05 + 1))
            if current_episode != 0 and current_episode % epsilon_update_interval == 0 and epsilon < max_epsilon:
                epsilon += 0.05
                # plot training episode where epsilon changes
                plt.axvline(x=current_episode, color="g",linestyle="--" )
                plt.annotate("Epsilon: {}".format(epsilon),(current_episode, 2.5 *  (current_episode/epsilon_update_interval)))
            
            accumulated_reward = 0
            is_terminal = False
            while(not is_terminal_state(train_type, time_elapsed, episode_time_limit, PURSUER_STUCK, EVADER_STUCK, PURSUER_STATE_DISCRETIZED["Opponent Position"])):
                time_elapsed = rospy.Time.now() - start_time
                
                # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 1.5 seconds
                if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                    PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=PURSUER_STATE_DISCRETIZED)    
                    del last_few_pursuer_positions[0]
                    
                if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                    EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= EVADER_STATE_DISCRETIZED)
                    del last_few_evader_positions[0]

                last_few_pursuer_positions.append(PURSUER_POSITION[:2])
                last_few_evader_positions.append(EVADER_POSITION[:2])
                
                # run opponent's decision-making in seperate thread
                thread = threading.Thread(target = follow_policy, args=(opponent_to_test, q_table_opponent))
                thread.start()

                # have current robot train using q-learning
                reward = q_learning_td(player_to_train, q_table_player, learning_rate= learning_rate, discount_factor= discount_factor, epsilon = epsilon,\
                    time_to_apply_action=time_to_apply_action)
                accumulated_reward += reward
                # follow_policy(opponent_to_test, q_table_opponent)
                thread.join()
            
            # keep track of how many times the pursuer managed to tag the evader
            if PURSUER_STATE_DISCRETIZED["Opponent Position"] == "Tagged":
                num_tagged += 1

            current_episode += 1
            training_reward += accumulated_reward
            
            if current_episode % 50 == 0:
                # plot learning curve using the average reward each 50 training episodes
                _plot_learning_curve(learning_curve,current_episode, training_reward/50)
                
                if training_reward > best_train_score:
                    # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                    # save q-table
                    with open("q_table_{}_best_training.txt".format(player_to_train), "w") as q_table_file:
                        q_table_file.seek(0)
                        q_table_file.write(pickle.dumps(q_table_player)) 
                    best_train_score = training_reward
                training_reward = 0
                
            if current_episode != 0 and current_episode % 500 == 0:
                # save the training curve figure every 500 episodes
                plt.savefig("td_curve_{}_episodes".format(current_episode), dpi=100)
            # save q-table
            with open("q_table_{}.txt".format(player_to_train), "w") as q_table_file:
                q_table_file.seek(0)
                q_table_file.write(pickle.dumps(q_table_player)) 
            rospy.loginfo("Saved Q-Table_{}".format(player_to_train))
            rospy.loginfo("Num tags = {}, Episodes so far {}".format(num_tagged, current_episode + 1))


def test(player_type, total_episodes = 2, episode_time_limit=30, time_to_apply_action = 0.33):
    current_episode = 0
    current_state = None
    accumulated_reward = 0 
    # keeps track of how many rounds the pursuer managed to tag the evader
    num_tagged = 0
    num_stuck = 0
    num_timeout = 0
    while(current_episode < total_episodes):
        # keep track of whether pursuer and evader are stuck, and what time
        rospy.loginfo("Testing episode {}".format(current_episode))
        global EVADER_STUCK
        global PURSUER_STUCK

        PURSUER_STUCK = False
        EVADER_STUCK = False

        last_few_pursuer_positions = []
        last_few_evader_positions = []

        # spawn at random points
        spawn_robots()

        while current_state is None or current_state["Opponent Position"] == "Tagged":
            if player_type == "pursuer":
                current_state = PURSUER_STATE_DISCRETIZED
                opponent_to_test = "evader"
                q_table_current = Q_TABLE_PURSUER
                q_table_opponent = Q_TABLE_EVADER
            else:
                current_state = EVADER_STATE_DISCRETIZED
                q_table_current = Q_TABLE_EVADER
                opponent_to_test = "pursuer"
                q_table_opponent = Q_TABLE_PURSUER
        
        # keeps track of how much time is left in current round
        start_time = rospy.Time.now()
        time_elapsed = rospy.Duration(secs=0)
        
        while(not is_terminal_state(train_type=player_type, time_elapsed=time_elapsed, \
                                    episode_time_limit=episode_time_limit,pursuer_stuck= PURSUER_STUCK, evader_stuck= EVADER_STUCK, \
                                    opponent_rating = current_state["Opponent Position"])):
            
            # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 2.50 seconds
            if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=PURSUER_STATE_DISCRETIZED)
                if PURSUER_STUCK:
                    num_stuck +=1    
                del last_few_pursuer_positions[0]
                
            if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=EVADER_STATE_DISCRETIZED)
                del last_few_evader_positions[0]

            last_few_pursuer_positions.append(PURSUER_POSITION[:2])
            last_few_evader_positions.append(EVADER_POSITION[:2])

            # run opponent's decision-making in seperate thread
            thread = threading.Thread(target = follow_policy, args=(opponent_to_test, q_table_opponent))
            thread.start()

            # follow Q-table to move robot
            follow_policy(player_type=player_type,q_table= q_table_current, time_to_apply_action=time_to_apply_action)
            
            # observe rewards of new state
            if player_type == "pursuer":
                current_state = PURSUER_STATE_DISCRETIZED
            else:
                current_state = EVADER_STATE_DISCRETIZED
            
            accumulated_reward += reward_function(player_type, current_state, verbose=False)

            if current_state["Opponent Position"] == "Tagged":
                num_tagged += 1

            time_elapsed = rospy.Time.now() - start_time
            if time_elapsed >= rospy.Duration(secs = episode_time_limit):
                num_timeout += 1
        current_episode += 1
        
    rospy.loginfo("TEST PHASE: {} times tagged out of {} testing rounds".format(num_tagged, total_episodes))
    rospy.loginfo("TEST PHASE: {} times stuck out of {} testing rounds".format(num_stuck, total_episodes))
    rospy.loginfo("TEST PHASE: {} times timeout out of {} testing rounds".format(num_timeout, total_episodes))

    return accumulated_reward / total_episodes

def load_q_table(q_table_name,player_type):
    if (not os.path.isfile(q_table_name)):
        rospy.loginfo("{} file is not found in current present working directory".format(q_table_name))
        return False
    else:
        with open(q_table_name, "rb") as q_table_file:
            if player_type == "pursuer": 
                global Q_TABLE_PURSUER
                Q_TABLE_PURSUER = pickle.load(q_table_file)
            else:
                global Q_TABLE_EVADER
                Q_TABLE_EVADER = pickle.load(q_table_file)
        return True

def main():
    rospy.loginfo("Waiting 3 seconds for everything to get set up...")
    rospy.sleep(3)
    rospy.init_node("robot_tag_node")
    global PURSUER_SCAN_SUBSCRIBER 
    global PURSUER_POSITION_SUBSCRIBER 
    global PURSUER_CMD_PUBLISHER 
    global EVADER_SCAN_SUBSCRIBER
    global EVADER_POSITION_SUBSCRIBER 
    global EVADER_CMD_PUBLISHER 
    # get pursuer's LaserScan reading to process and yield pursuer's current state
    PURSUER_SCAN_SUBSCRIBER = rospy.Subscriber("/pursuer/scan", LaserScan, callback=get_current_state, callback_args={"player_type" : "pursuer" ,"verbose": False})
    # repeat with evader
    EVADER_SCAN_SUBSCRIBER = rospy.Subscriber("/evader/scan", LaserScan, callback=get_current_state, callback_args={"player_type" : "evader" ,"verbose": False})

    # get players 's positions
    PURSUER_POSITION_SUBSCRIBER = rospy.Subscriber("/pursuer/odom", Odometry, callback = get_robot_location, callback_args = {"player_type": "pursuer", "verbose": False})
    EVADER_POSITION_SUBSCRIBER = rospy.Subscriber("/evader/odom", Odometry, callback = get_robot_location, callback_args = {"player_type": "evader", "verbose": False})

    # different cmd publishers to contorl pursuers and evader robots differently
    PURSUER_CMD_PUBLISHER = rospy.Publisher("pursuer/cmd_vel", Twist, latch=True, queue_size=1)
    EVADER_CMD_PUBLISHER = rospy.Publisher("evader/cmd_vel", Twist, latch=True, queue_size=1)
    

    for player_type in ["pursuer", "evader"]:
        if (not os.path.isfile("q_table_{}.txt".format(player_type))):
            rospy.loginfo("Created and initialized a new Q_Table for {}".format(player_type))
            create_q_table(player_type)
        with open("q_table_{}.txt".format(player_type), "rb") as q_table_file:
            if player_type == "pursuer": 
                global Q_TABLE_PURSUER
                Q_TABLE_PURSUER = pickle.load(q_table_file)
            else:
                global Q_TABLE_EVADER
                Q_TABLE_EVADER = pickle.load(q_table_file)
    
    train(train_type = "pursuer", starting_epsilon=0.1, total_episodes=35000, episode_time_limit=45)
    # replace_speed_in_q_table("q_table_pursuer.txt",0.1,0.075)

    # successfully_loaded = load_q_table(q_table_name="q_table_pursuer_best_testing_new_state_design_30k_55%_tag_rate.txt", player_type="pursuer")
    # if successfully_loaded:
    #     test("pursuer", total_episodes= 100, episode_time_limit=60)
    
    
    # rospy.spin()

 

if __name__ == "__main__":
    main()