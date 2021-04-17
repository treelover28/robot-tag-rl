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
PURSUER_MEAN_DISTANCE_TO_OBSTACLE = None 
DISTANCE_BETWEEN_PLAYERS = None
LASERSCAN_MAX_RANGE = None 


STARTING_LOCATIONS = [(0.5,0), (-0.5,0), (-0.5,1), (0.5,1), \
    (-0.5,-1), (-0.5, -1), (2, 0), (-2,0), (-1,2), (-1,-2), \
    (1.8,1), (2,-1)]
# Gameplay hyperparameters
TIMEOUT = False
GAME_TIME = 30 # a round/traning episode last maximum 30 seconds
# State Space Hyperparameters
SAFE_DISTANCE_FROM_OBSTACLE = 0.3
ROTATIONAL_ACTIONS = [45,20,0,-20,-45]
TRANSLATION_SPEED = 0.3 
DIRECTIONAL_STATES = ["Front", "Left", "Right", "Opponent Position"]
FRONT_RATINGS = ["Close", "OK", "Far"]
LEFT_RATINGS = ["Too Close", "Close", "OK", "Far", "Too Far"]
RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far", "Too Far"]
OPPONENT_RATINGS = ["Close Left", "Left", "Close Front", "Front", "Right", "Close Right", "Bottom", "Close Bottom", "Tagged"]

# DIFFERENT TOPICS SUBSCRIBERS AND LISTENERS
PURSUER_SCAN_SUBSCRIBER = None 
PURSUER_POSITION_SUBSCRIBER = None 
PURSUER_CMD_PUBLISHER = None 
EVADER_SCAN_SUBSCRIBER = None 
EVADER_POSITION_SUBSCRIBER = None 
EVADER_CMD_PUBLISHER = None 
TRANSFORM_LISTENER = None
TF_BUFFER = None

# Q-tables
Q_TABLE_PURSUER = None 
Q_TABLE_EVADER = None 

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def reward_function(player_type, state):
    global DISTANCE_BETWEEN_PLAYERS
    DISTANCE_BETWEEN_PLAYERS = np.linalg.norm(np.array(PURSUER_POSITION[0:2]) - np.array(EVADER_POSITION[0:2]))
    
    if player_type == "pursuer":
        # if the pursuer gets stuck, it loses that game -> negative reward
        if PURSUER_STUCK:
            reward = -2
        elif state["Opponent Position"] == "Tagged":
            reward = 5 
        # to avoid obstacles 
        elif (state["Left"] in ["Close", "Too Close"] or \
            state["Right"] in ["Close", "Too Close"] or \
            state["Front"] in ["Close"]) and \
            (DISTANCE_BETWEEN_PLAYERS > PURSUER_MEAN_DISTANCE_TO_OBSTACLE):
            rospy.loginfo("Obstacle is nearby and evader is far")
            reward = -0.5 - sigmoid(1/DISTANCE_BETWEEN_PLAYERS)
        elif((state["Left"] in ["Close", "Too Close"] or \
              state["Right"] in ["Close", "Too Close"] or \
              state["Front"] in ["Close"]) and DISTANCE_BETWEEN_PLAYERS <= PURSUER_MEAN_DISTANCE_TO_OBSTACLE
            ) or \
            ((DISTANCE_BETWEEN_PLAYERS <= SAFE_DISTANCE_FROM_OBSTACLE * 1.2) or \
            state["Opponent Position"] in ["Close Left", "Close Front", "Close Bottom", "Close Right"]):
            rospy.loginfo("Evader is nearby")
            reward = sigmoid(1/DISTANCE_BETWEEN_PLAYERS) * 2.5 
        elif state["Opponent Position"] == "Front":
            reward = 0.5
        # there is no obstacle nearby and the target evader is far away
        elif DISTANCE_BETWEEN_PLAYERS >= PURSUER_MEAN_DISTANCE_TO_OBSTACLE:
            rospy.loginfo("No obstacle nearby and evader is far away")
            reward = -0.1 - sigmoid(1/DISTANCE_BETWEEN_PLAYERS)
        else:
            reward = 0
    elif player_type == "evader":
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
        elif DISTANCE_BETWEEN_PLAYERS > PURSUER_MEAN_DISTANCE_TO_OBSTACLE:
            reward = sigmoid(DISTANCE_BETWEEN_PLAYERS)
        # punish states where the tagee lets the pursuer gets too close 
        elif (DISTANCE_BETWEEN_PLAYERS <= SAFE_DISTANCE_FROM_OBSTACLE * 1.3) or \
             state["Opponent Position"] in ["Close Left", "Close Front", "Close Bottom", "Close Right"]:
            reward = -2.5 * sigmoid(1/DISTANCE_BETWEEN_PLAYERS)
        else:
            reward = 0 
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
    if 0 <= angle_deg < 45 or 315 <= angle_deg < 360:
       direction_rating = "Front"
    elif 45 <= angle_deg < 135:
        direction_rating = "Left"
    elif 225 <= angle_deg < 315:
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
    elif direction in ["Left", "Right"]:
        interval = SAFE_DISTANCE_FROM_OBSTACLE/2.5
        if distance <= interval * 1.5:
            rating = "Too Close"
        elif distance <= (interval * 2.2):
            rating = "Close"
        elif distance <= (interval * 3.4):
            rating = "OK"
        elif distance <= (interval * 4.5):
            rating = "Far"
        else:
            rating = "Too Far"
    return rating


def get_current_state(message,args):
    ranges_data = message.ranges

    global LASERSCAN_MAX_RANGE
    LASERSCAN_MAX_RANGE = message.range_max 


    player_type = args["player_type"]
    verbose = args["verbose"]

    front_sector = range(0,45) + range(315,360)
    left_sector = range(45,135) 
    right_sector = range(225,315) 

    # use the smallest distance detected at each directional state
    min_front, min_left, min_right = [float("inf") for i in range(0,3)]

    for angle in front_sector:
        min_front = min(min_front, ranges_data[angle])
    for angle in left_sector:
        min_left = min(min_left, ranges_data[angle])
    for angle in right_sector:
        min_right = min(min_right, ranges_data[angle])

    if player_type == "pursuer":
        global PURSUER_STATE_DISCRETIZED 
        
        PURSUER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front), \
            "Left" : get_distance_rating("Left", min_left), \
            "Right": get_distance_rating("Right", min_right), \
            "Opponent Position": get_opponent_position_rating(PURSUER_POSITION, EVADER_POSITION)
        }

        global PURSUER_MEAN_DISTANCE_TO_OBSTACLE 
        PURSUER_MEAN_DISTANCE_TO_OBSTACLE= sum([min_front, min_left, min_right])/3
        
        if verbose:
            rospy.loginfo("Pursuer's state: {}".format(PURSUER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of pursuer's state: {}".format(reward_function("pursuer", PURSUER_STATE_DISCRETIZED)))
    else:
        global EVADER_STATE_DISCRETIZED 
        EVADER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front), \
            "Left" : get_distance_rating("Left", min_left), \
            "Right": get_distance_rating("Right", min_right), \
            "Opponent Position": get_opponent_position_rating( EVADER_POSITION, PURSUER_POSITION)
        }

        global EVADER_DISTANCE_FROM_NEAREST_OBSTACLE 
        EVADER_DISTANCE_FROM_NEAREST_OBSTACLE= min(min_front, min_left, min_right)

        if verbose:
            rospy.loginfo("Evader's state: {}".format(EVADER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of evader's state: {}".format(reward_function("evader", EVADER_STATE_DISCRETIZED)))

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
    
    q_table = {}
    all_states = []
    _get_state_permutations(0, list(), 4, all_states)

    for state in all_states:
        # unpack tuple to get corresponding distance ratings for each direction
        front_rating, left_rating, right_rating, opponent_rating = state
        state_dictionary = ({"Front": front_rating,  "Left": left_rating, "Right": right_rating, "Opponent Positon": opponent_rating})
        # convert state (originally a list) to a tuple which is hashable 
        # and could be stored in the Q_table which is a dictionary
        state_tuple = tuple(state)

        # initialize the q-value for each action to be 0
        q_values = {}
        for action in ROTATIONAL_ACTIONS:
            q_values[action] = 0

        # each state has n q-values associated with n actions that can be done, plus
        q_table[state_tuple] = q_values
    
    # save Q-table as an external file
    with open("q_table_{}.txt".format(player_type), "w") as q_table_file:
        q_table_file.seek(0)
        q_table_file.write(pickle.dumps(q_table)) 

def _get_state_permutations(current_list_index, prefix, k, states_accumulator):
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
    
    if current_list_index == 0:
        rating_list = FRONT_RATINGS
    elif current_list_index == 1:
        rating_list = LEFT_RATINGS
    elif current_list_index == 2:
        rating_list = RIGHT_RATINGS
    else:
        rating_list = OPPONENT_RATINGS

    for i in range(len(rating_list)):
        new_prefix = (prefix + [rating_list[i]])
        _get_state_permutations(current_list_index + 1, new_prefix, k-1, states_accumulator)

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
        if (r < epsilon):
            max_q_values = float("-inf")
            for action in ROTATIONAL_ACTIONS:
                if q_values[action] > max_q_values:
                    chosen_action = action
                    max_q_values = q_values[action]
        else:
            chosen_action = ROTATIONAL_ACTIONS[int(random.random() * len(q_values))]
            exploration_flag = True 

        if verbose:
            if exploration_flag:
                rospy.loginfo("Exploration. Random action chosen.")
            else:
                rospy.loginfo("Exploitation. Choose action with max-utility.")
            rospy.loginfo("Action: angular {})".format(chosen_action))
        return TRANSLATION_SPEED, chosen_action
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
    rospy.loginfo("{}_current State: {}".format(player_type , current_state))
    # get action A from S using policy
    translation_speed, turn_action = get_policy(q_table, current_state, verbose = False, epsilon= epsilon)
    # take action A and move player, this would change the player's state
    # rospy.loginfo("Chosen action {}".format(action))
    move_robot(player_type, translation_speed, turn_action)
    # give the robot some time to apply action => proper state transition
    rospy.sleep(time_to_apply_action)
    
    # robot is now in new state S' and observe reward R(S') 
    new_state = PURSUER_STATE_DISCRETIZED if (player_type == "pursuer") else EVADER_STATE_DISCRETIZED
    reward = reward_function(player_type, new_state)
    rospy.loginfo("{}'s reward: {}".format(player_type, reward))
    
    # update Q-value for Q(S,A)
    # Q(S,A) = Q(S,A) +  learning_rate*(reward + discount_factor* (argmax_A' Q(S', A')) - Q(S,A))
    current_state_tuple = get_state_tuple_from_dictionary(current_state)
    new_state_tuple = get_state_tuple_from_dictionary(new_state)

    translation_speed, best_action = get_policy(q_table, new_state, verbose=False, epsilon = 1.0)
    q_table[current_state_tuple][turn_action] += learning_rate*(reward + \
                    discount_factor * q_table[new_state_tuple][best_action] - q_table[current_state_tuple][turn_action])    
    
    return reward


def random_walk_behavior(robot_type, robot_state, random_action_chance = 0.2):
    if robot_type == "evader":
        is_stuck = EVADER_STUCK
    else:
        is_stuck = PURSUER_STUCK

    if robot_state["Front"] == "Close" and robot_state["Left"] == "Very Close" and robot_state["Right"] == "Very Close" and is_stuck:
        translation_speed = -0.1 
        turn_angle = -60
    elif robot_state["Front"] == "Close" and is_stuck:
        translation_speed = -1*TRANSLATION_SPEED
        turn_angle = -60
    else:
        translation_speed = TRANSLATION_SPEED   
        # 20% chance of making random turns
        if (random.random() < 0.2):
            turn_angle = random.randint(0,359)
        else:
            # just go straight else
            turn_angle = 0 
    return (translation_speed, turn_angle)


def follow_policy(player_type, q_table):

    if player_type == "pursuer":
        current_state = PURSUER_STATE_DISCRETIZED
        translation_speed, turn_action = get_policy(q_table, current_state, verbose= False, epsilon = 1.0)
    elif player_type == "evader":
        current_state = EVADER_STATE_DISCRETIZED
        translation_speed, turn_action = random_walk_behavior(robot_type="evader", robot_state=current_state)
    
    move_robot(player_type, translation_speed, turn_action)
    rospy.sleep(0.33)

def is_stuck(last_few_positions):
    # rospy.loginfo("Checking is stuck")
    if last_few_positions is not None:
        changes_in_x = 0
        changes_in_y = 0
        for i in range(1,len(last_few_positions)):
            changes_in_x += abs(last_few_positions[i][0] - last_few_positions[i - 1][0])
            changes_in_y += abs(last_few_positions[i][1] - last_few_positions[i - 1][1])
        # if accumulated changes in both coordinates are less than a very small number, 
        # the robot is probably stuck
        return changes_in_x < 0.05 and changes_in_y < 0.05
    return False

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
    plt.ylim(-50, 200)
    plt.legend(loc="upper left")
    plt.axhline(y= 0, color = "g", linestyle = "-")
    plt.show(block=False)

    training_rewards = []
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
                test_reward = test(player_to_train)
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
                plt.annotate("Epsilon: {}".format(epsilon),(current_episode, 80 + 5 *  (current_episode/epsilon_update_interval)))
            
            accumulated_reward = 0
            is_terminal = False
            while(not is_terminal_state(train_type, time_elapsed, episode_time_limit, PURSUER_STUCK, EVADER_STUCK, PURSUER_STATE_DISCRETIZED["Opponent Position"])):
                time_elapsed = rospy.Time.now() - start_time
                
                # check if robots are stuck
                if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                    PURSUER_STUCK = is_stuck(last_few_pursuer_positions)    
                    del last_few_pursuer_positions[0]
                    
                if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                    EVADER_STUCK = is_stuck(last_few_evader_positions)
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
            
            current_episode += 1
           
            training_rewards.append(accumulated_reward)
            
            if current_episode % 20 == 0:
                # plot learning curve using the average reward each 20 training episodes
                _plot_learning_curve(learning_curve,current_episode, sum(training_rewards)/len(training_rewards))
                training_rewards = []
        
            if current_episode != 0 and current_episode % 500 == 0:
                # save the training curve figure every 500 episodes
                plt.savefig("td_curve_{}_episodes".format(current_episode), dpi=100)
            # save q-table
            with open("q_table_{}.txt".format(player_to_train), "w") as q_table_file:
                q_table_file.seek(0)
                q_table_file.write(pickle.dumps(q_table_player)) 
            rospy.loginfo("Saved Q-Table_{}".format(player_to_train))


def test(player_type, total_episodes = 2, episode_time_limit=30, time_to_apply_action = 0.33):
    current_episode = 0

    
    current_state = None
    accumulated_reward = 0 
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

        start_time = rospy.Time.now()
        time_elapsed = rospy.Duration(secs=0)
        while(not is_terminal_state(train_type=player_type, time_elapsed=time_elapsed, \
                                    episode_time_limit=episode_time_limit,pursuer_stuck= PURSUER_STUCK, evader_stuck= EVADER_STUCK, \
                                    opponent_rating = current_state["Opponent Position"])):
            time_elapsed = rospy.Time.now() - start_time
            # check if robots are stuck
            if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                PURSUER_STUCK = is_stuck(last_few_pursuer_positions)    
                del last_few_pursuer_positions[0]
                
            if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                EVADER_STUCK = is_stuck(last_few_evader_positions)
                del last_few_evader_positions[0]

            last_few_pursuer_positions.append(PURSUER_POSITION[:2])
            last_few_evader_positions.append(EVADER_POSITION[:2])

            # run opponent's decision-making in seperate thread
            thread = threading.Thread(target = follow_policy, args=(opponent_to_test, q_table_opponent))
            thread.start()

            # get action A from S using policy
            translation_speed, turn_action = get_policy(q_table_current, current_state, verbose = False, epsilon= 1)
            # take action A and move player, this would change the player's state
            # rospy.loginfo("Chosen action {}".format(action))
            move_robot(player_type, translation_speed, turn_action)
            # give the robot some time to apply action => proper state transition
            rospy.sleep(time_to_apply_action)
            # observe rewards of new state
            if player_type == "pursuer":
                current_state = PURSUER_STATE_DISCRETIZED
            else:
                current_state = EVADER_STATE_DISCRETIZED
            accumulated_reward += reward_function(player_type, current_state)
        current_episode += 1
    return accumulated_reward / total_episodes



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
    global TRANSFORM_LISTENER
    global TF_BUFFER
    # get pursuer's LaserScan reading to process and yield pursuer's current state
    PURSUER_SCAN_SUBSCRIBER = rospy.Subscriber("/pursuer/scan", LaserScan, callback=get_current_state, callback_args={"player_type" : "pursuer" ,"verbose": True})
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
    # rospy.spin()
    # train(train_type = "pursuer", starting_epsilon=0.4, total_episodes=2500)
    test("pursuer", total_episodes= 10)


if __name__ == "__main__":
    main()