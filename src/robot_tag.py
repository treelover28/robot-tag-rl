#! /usr/bin/python

import rospy
import os.path
import cPickle as pickle
import tf
import sys 

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from math import pi as PI
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


TAGGER_STATE_DISCRETIZED = None 
TAGGER_STATE_CONTINUOUS = None
TAGGER_POSITION = None 
TAGGER_STUCK = False

TAGGEE_STATE_DISCRETIZED = None 
TAGGEE_STATE_CONTINUOUS = None
TAGGEE_STUCK = False
TAGGEE_POSITION = None 

# Gameplay hyperparameters
TIMEOUT = False
GAME_TIME = 30 # a round/traning episode last maximum 30 seconds

# State Space Hyperparameters
DISTANCE_FROM_OBSTACLE = 0.3
ROTATIONAL_ACTIONS = [45,20,0,-20,-45]
TRANSLATION_SPEED = 0.2 
DIRECTIONAL_STATES = ["Front", "Left", "Right", "Opponent Position"]
FRONT_RATINGS = ["Close", "OK", "Far"]
LEFT_RATINGS = ["Too Close", "Close", "OK", "Far", "Too Far"]
RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far", "Too Far"]
OPPONENT_RATINGS = ["Close Left", "Left", "Close Front", "Front", "Right", "Close Right", "Bottom", "Close Bottom", "Tagged"]

# DIFFERENT TOPICS SUBSCRIBERS AND LISTENERS
TAGGER_SCAN_SUBSCRIBER = None 
TAGGER_POSITION_SUBSCRIBER = None 
TAGGER_CMD_PUBLISHER = None 
TAGGEE_SCAN_SUBSCRIBER = None 
TAGGEE_POSITION_SUBSCRIBER = None 
TAGGEE_CMD_PUBLISHER = None 
TRANSFORM_LISTENER = None
TF_BUFFER = None

# Q-tables
Q_TABLE_TAGGER = None 
Q_TABLE_TAGGEE = None 

def reward_function(player_type, state):
    distance = np.linalg.norm(np.array(TAGGER_POSITION[0:2]) - np.array(TAGGEE_POSITION[0:2]))
    print(distance)
    if player_type == "tagger":
        if TAGGER_STUCK:
            reward = -2
        elif TIMEOUT: 
            reward = -2
        # to avoid obstacles if target is far away
        elif (state["Left"] in ["Close", "Too Close"] or \
           state["Right"] in ["Close", "Too Close"] or \
           state["Front"] in ["Close"]) and distance > DISTANCE_FROM_OBSTACLE * 1.2:
           reward = -1
        elif distance > DISTANCE_FROM_OBSTACLE * 1.2:
            reward = -0.25
        # encourage states where the tagger is close to the taggee 
        elif state["Opponent Position"] in ["Close Left", "Close Front", "Close Bottom", "Close Right"]:
            reward = 1
        # tagger wins if it successfully tagged the taggee
        elif state["Opponent Position"] == "Tagged":
            reward = 2
        else:
            reward = 0 
    elif player_type == "taggee":
        if TAGGEE_STUCK:
            reward = -2
        elif TIMEOUT:
            # if tagger hasn't caught taggee before the round ends, the taggee wins and gets rewarded 
            reward = 2 
        # to also promote obstacle avoidance
        elif (state["Left"] in ["Close", "Too Close"] or \
           state["Right"] in ["Close", "Too Close"] or \
           state["Front"] in ["Close"]) and distance > DISTANCE_FROM_OBSTACLE * 1.2:
           reward = -1
        elif distance > DISTANCE_FROM_OBSTACLE * 1.2:
            reward = 0.25
        # punish states where the tagee lets the tagger gets too close 
        elif state["Opponent Position"] in ["Close Left", "Close Front", "Close Bottom", "Close Right"]:
            reward = -1
        # if the taggee gets tagged, it has lost the game
        elif state["Opponent Position"] == "Tagged":
            reward = -2
        else:
            reward = 0 
    return reward 

def get_opponent_position_rating(player_A, player_B):
    player_A_position = np.array(player_A[:3])
    player_A_orientation = np.array(player_A[2:])
    player_B_position = np.array(player_B[:3])
    player_B_orientation = np.array(player_B[2:])
    plane_normal = np.array([0,0,1])

    player_A_yaw = tf.transformations.euler_from_quaternion(player_A_orientation)[2]
    
    vector_A = np.array([np.cos(player_A_yaw), np.sin(player_A_yaw), 0])
    vector_B = player_B_position - player_A_position
    
    # print("Player A = {} and player B = {}".format(player_A_position, player_B_position))
    # print("vector A = {} and vector B = {}".format(vector_A, vector_B))
    
    dot_product = np.dot(vector_A, vector_B)
    # print("dot product - direction: {}".format(dot_product))
    norm_A = np.linalg.norm(vector_A)
    norm_B = np.linalg.norm(vector_B)
    angle_rad = np.arccos(dot_product/(np.dot(norm_A,norm_B)))
    cross = np.cross(vector_A, vector_B)
    if (np.dot(plane_normal, cross) < 0):
        angle_rad *= -1 
    angle_deg =  np.rad2deg(angle_rad) 
    # print("angle: {}".format(angle_deg))
    distance = np.linalg.norm(vector_B)
    # print("distance: {}".format(distance))

    if distance <= 0.1:
        return "Tagged"
    if 45 <= angle_deg < 135:
       direction_rating = "Front"
    elif 135 <= angle_deg < 180 or -180 <= angle_deg < -135:
        direction_rating = "Left"
    elif 0 <= angle_deg < 45 or -45 <= angle_deg < 0:
        direction_rating = "Right"
    else:
        direction_rating = "Bottom"

    distance_rating = ""
    if distance <= DISTANCE_FROM_OBSTACLE * 1.2:
        distance_rating = "Close"
    
    return (distance_rating + " " + direction_rating).strip()

def get_distance_rating(direction, distance):
    if direction == "Front":
        interval = DISTANCE_FROM_OBSTACLE
        if distance <= (interval * 1.5):
            rating = "Close"
        else:
            rating = "Far"
    elif direction in ["Left", "Right"]:
        interval = DISTANCE_FROM_OBSTACLE/2.5
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

    if player_type == "tagger":
        global TAGGER_STATE_DISCRETIZED 
        TAGGER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front), \
            "Left" : get_distance_rating("Left", min_left), \
            "Right": get_distance_rating("Right", min_right), \
            "Opponent Position": get_opponent_position_rating(TAGGER_POSITION, TAGGEE_POSITION)
        }

        if verbose:
            rospy.loginfo("Tagger's state: {}".format(TAGGER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of tagger's state: {}".format(reward_function("tagger", TAGGER_STATE_DISCRETIZED)))
    else:
        global TAGGEE_STATE_DISCRETIZED 
        TAGGEE_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front), \
            "Left" : get_distance_rating("Left", min_left), \
            "Right": get_distance_rating("Right", min_right), \
            "Opponent Position": get_opponent_position_rating( TAGGEE_POSITION, TAGGER_POSITION)
        }

        if verbose:
            rospy.loginfo("Taggee's state: {}".format(TAGGEE_STATE_DISCRETIZED))
            rospy.loginfo("Reward of taggee's state: {}".format(reward_function("taggee", TAGGEE_STATE_DISCRETIZED)))

def get_robot_location(message, args):
    player_type = args["player_type"]
    verbose = args["verbose"]
    
    position = [message.pose.pose.position.x, message.pose.pose.position.y, 0, \
         message.pose.pose.orientation.x, message.pose.pose.orientation.y,message.pose.pose.orientation.z, message.pose.pose.orientation.w]
    if player_type == "tagger":
        global TAGGER_POSITION
        TAGGER_POSITION = position
        if verbose:
            rospy.loginfo("Tagger's position is {}".format(TAGGER_POSITION))
    else:
        global TAGGEE_POSITION
        TAGGEE_POSITION = position
        if verbose:
            rospy.loginfo("Taggee's position is {}".format(TAGGEE_POSITION))

def create_q_table(player_type = "tagger"):
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
        
        
def main():
    rospy.sleep(1)
    rospy.init_node("robot_tag_node")
    global TAGGER_SCAN_SUBSCRIBER 
    global TAGGER_POSITION_SUBSCRIBER 
    global TAGGER_CMD_PUBLISHER 
    global TAGGEE_SCAN_SUBSCRIBER
    global TAGGEE_POSITION_SUBSCRIBER 
    global TAGGEE_CMD_PUBLISHER 
    global TRANSFORM_LISTENER
    global TF_BUFFER
    # get tagger's LaserScan reading to process and yield tagger's current state
    TAGGER_SCAN_SUBSCRIBER = rospy.Subscriber("/tagger/scan", LaserScan, callback=get_current_state, callback_args={"player_type" : "tagger" ,"verbose": False})
    # repeat with taggee
    TAGGEE_SCAN_SUBSCRIBER = rospy.Subscriber("/taggee/scan", LaserScan, callback=get_current_state, callback_args={"player_type" : "taggee" ,"verbose": True})

    # get players 's positions
    TAGGER_POSITION_SUBSCRIBER = rospy.Subscriber("/tagger/odom", Odometry, callback = get_robot_location, callback_args = {"player_type": "tagger", "verbose": False})
    TAGGEE_POSITION_SUBSCRIBER = rospy.Subscriber("/taggee/odom", Odometry, callback = get_robot_location, callback_args = {"player_type": "taggee", "verbose": False})

    # different cmd publishers to contorl taggers and taggee robots differently
    TAGGER_CMD_PUBLISHER = rospy.Publisher("tagger/cmd_vel", Twist, latch=True, queue_size=1)
    TAGGEE_CMD_PUBLISHER = rospy.Publisher("taggee/cmd_vel", Twist, latch=True, queue_size=1)
    

    for player_type in ["tagger", "taggee"]:
        if (not os.path.isfile("q_table_{}.txt".format(player_type))):
            rospy.loginfo("Created and initialized a new Q_Table for {}".format(player_type))
            create_q_table(player_type)
        with open("q_table_{}.txt".format(player_type), "rb") as q_table_file:
            if player_type == "tagger": 
                global Q_TABLE_TAGGER
                Q_TABLE_TAGGER = pickle.load(q_table_file)
            else:
                global Q_TABLE_TAGGEE
                Q_TABLE_TAGGEE = pickle.load(q_table_file)
    
    rate = rospy.Rate(30)
    while (not rospy.is_shutdown()):
        rate.sleep()
        pass 



if __name__ == "__main__":
    main()