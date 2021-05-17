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



GAME_TIMEOUT = False


PURSUER_STATE_DISCRETIZED = None 
PURSUER_STATE_CONTINUOUS = None
PURSUER_POSITION = None 
PURSUER_STUCK = False

EVADER_STATE_DISCRETIZED = None 
EVADER_STATE_CONTINUOUS = None
EVADER_STUCK = False
EVADER_POSITION = None 

EVADER_MIN_DISTANCE_TO_OBSTACLE = None
PURSUER_MIN_DISTANCE_TO_OBSTACLE = None 
DISTANCE_BETWEEN_PLAYERS = None

RESCUE_PURSUER_FAILED = False
RESCUE_EVADER_FAILED = False
# for ros_plaza
STARTING_LOCATIONS = [(0,1.2), (-2,1), (0,-1), (0,1.5), (0,-2), (-2,-1), (0.5,0), (-2,1.8),(1,0), (1,-2)]

# for ros pillars map
# STARTING_LOCATIONS = [(0,1), (-1,0), (0,-1), (1,0), (-1,-2), (-1,2)]
# for original ros map with all the pillars 
# STARTING_LOCATIONS = [(0.5,-0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (-1,-2), (-1,2)]

# State Space Hyperparameters
SAFE_DISTANCE_FROM_OBSTACLE = 0.3
ROTATIONAL_ACTIONS = [60,40, 20, 5, 0, -5, -20,-40,-60]
# slow speed 0.1 to help it slow down when near obstacle
# regular speed is 0.2 
# negative velocity is to help it reverse and rescue itself whenever it gets stuck
TRANSLATION_SPEED = [0.1, 0.2]
DIRECTIONAL_STATES = ["Front", "Upper Left", "Upper Right", "Lower Left", "Lower Right","Opponent Position"]
FRONT_RATINGS = ["Close", "OK", "Far"]
UPPER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
UPPER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
LOWER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
LOWER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
OPPONENT_RATINGS = ["Close Left", "Left", "Close Front", "Front", "Right", "Close Right", "Bottom", "Close Bottom", "Tagged"]

PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION = ""
EVADER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION = ""

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

# WIDTH OF TURTLEBOTS:
PURSUER_RADIUS = 1.0/8 # 4 Waffles one unit width wise
EVADER_RADIUS = 1.0/8 # 2 Burgers is roughly 1 waffle width wise

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def reward_function(player_type, state, verbose = True):
    global DISTANCE_BETWEEN_PLAYERS
    DISTANCE_BETWEEN_PLAYERS = np.linalg.norm(np.array(PURSUER_POSITION[0:2]) - np.array(EVADER_POSITION[0:2]))
    
    if player_type == "pursuer":
        # if the pursuer gets stuck, it loses that game -> negative reward
        # the negative reward is also based on how badly it lost that round

        # the distance is between player calculated from the positions is the distance from one's robot's center to another robot's center
        # while distance gathered from the pursuer's LIDAR is from the pursuer's center to the evader's nearest SIDE 
        # thus, we need to adjust this DISTANCE_BETWEEN_PLAYERS by the evader's radius to better compare the two 
        TRUE_DISTANCE_BETWEEN_PLAYERS = (DISTANCE_BETWEEN_PLAYERS - EVADER_RADIUS)
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = (PURSUER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE)

        # rospy.loginfo("PURSUER_MIN_DISTANCE: {}\nTRUE_DISTANCE_BETWEEN_PLAYERS: {}\nTRUE_SAFE_DISTANCE_FROM_OBSTACLE: {}".format(PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_DISTANCE_BETWEEN_PLAYERS ,TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
        if PURSUER_STUCK:
            # rospy.loginfo("STUCK!")
            state_description = "STUCK!"
            reward = -30
        elif state["Opponent Position"] == "Tagged":
            # rospy.loginfo("TAGGED!")
            state_description = "TAGGED!"
            reward = 30 

        # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
        elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
            and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
            and state["Front"] in ["Close", "OK"] and state["Opponent Position"] in ["Front", "Close Front"]:
            
            state_description = "Obstacle on both sides, but there is opening in front and the target is in front nearby"
            reward = -sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE) + 2*sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS)
        
        # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
        elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
            and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
            and state["Front"] == "Far" and state["Opponent Position"] == "Front":
            
            state_description = "Obstacle on both sides, but there is opening in front and the target is in front far away"
            reward = -sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE) + sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS) 
        
        # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
        elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
            and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
            and state["Front"] != "Close":

            state_description = "Obstacle on both sides, but there is opening in front but opponent is not in front"
            reward = -sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE) - sigmoid(TRUE_DISTANCE_BETWEEN_PLAYERS) 
        
        # if there are obstacles nearby ON ONE SIDE(that is not the evader), and the evader is far away, promote obstacles avoidance behavior
        elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left")  or \
               (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right") or \
               (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left") or \
               (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right"))
            ) and TRUE_DISTANCE_BETWEEN_PLAYERS > TRUE_SAFE_DISTANCE_FROM_OBSTACLE\
              and PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION != "Front": 

            state_description = "Obstacle is a lot nearer on the sides compared to evader. Prioritize obstacle avoidance"
            # extra punishment depending on how far the evader is and how close the pursuer is to an obstacle
            reward = -0.5 - sigmoid(TRUE_DISTANCE_BETWEEN_PLAYERS) - sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE)
        
        # there is an obstacle in front that is not the opponent
        elif (state["Front"] == "Close" or (PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION == "Front" and PURSUER_MIN_DISTANCE_TO_OBSTACLE <= TRUE_SAFE_DISTANCE_FROM_OBSTACLE))\
            and state["Opponent Position"] not in  ["Front", "Close Front"]:
            
            state_description = "Obstacle directly infront that is not the opponent. Prioritize obstacle avoidance"
            reward = -0.25 - sigmoid(TRUE_DISTANCE_BETWEEN_PLAYERS) - sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE)
        
        # check for special case where opponent is directly in front, yet behind an obstacle, so robot priotize obstacle avoidance
        elif state["Front"] == "Close" and state["Opponent Position"] == "Front" \
            and TRUE_DISTANCE_BETWEEN_PLAYERS >= PURSUER_MIN_DISTANCE_TO_OBSTACLE\
            and PURSUER_MIN_DISTANCE_TO_OBSTACLE < TRUE_SAFE_DISTANCE_FROM_OBSTACLE\
            and PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION == "Front":

            state_description = "Evader is directly in front, but probably is behind an obstacle. Prioritize obstacle avoidance"
            reward = -0.25 - sigmoid(TRUE_DISTANCE_BETWEEN_PLAYERS) - sigmoid(1/PURSUER_MIN_DISTANCE_TO_OBSTACLE)
        
        # else if the evader is in front and closeby, and we are relatively safe from obstacles on either sides
        elif state["Opponent Position"] == "Front" and TRUE_DISTANCE_BETWEEN_PLAYERS <= 1.0 and\
            TRUE_DISTANCE_BETWEEN_PLAYERS <= PURSUER_MIN_DISTANCE_TO_OBSTACLE:
            
            # encourage robot to orient itself such that the opponent is directly in front of it
            # take away the sigmoid of the distance to encourage it to minimize such distance 
            state_description = "Evader is in front and close enough by, and we are relatively safe from obstacle!"
            reward = sigmoid(1.0/TRUE_DISTANCE_BETWEEN_PLAYERS)
        
        elif state["Opponent Position"] == "Front":
            
            state_description = "Evader is in front but not that close"
            reward = 0.5* sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS)
        
        # if the other robot is nearby and there is an obstacle, there is a chance that obstacle 
        # may be the other robot, so we encourage those states
        # or if the distance between players are very close
        elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left")  or \
               (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
               (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left") or \
               (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
               (state["Front"] in ["Close"] and state["Opponent Position"] == "Close Front")) and\
                   TRUE_DISTANCE_BETWEEN_PLAYERS <= PURSUER_MIN_DISTANCE_TO_OBSTACLE) or\
                       (TRUE_DISTANCE_BETWEEN_PLAYERS <= TRUE_SAFE_DISTANCE_FROM_OBSTACLE):
            state_description = "Evader is nearby and we are relatively safe from obstacles"
            reward = 2.5 * sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS)
    
        # there is no obstacle nearby and the target evader is far away
        elif state["Upper Left"] not in ["Close", "Too Close"] and state["Lower Left"] not in ["Close", "Too Close"]\
            and state["Upper Right"] not in ["Close", "Too Close"] and state["Lower Right"] not in ["Close", "Too Close"]\
            and state["Front"] != "Close" and TRUE_DISTANCE_BETWEEN_PLAYERS >= SAFE_DISTANCE_FROM_OBSTACLE:
            
            state_description = "Safe from obstacle but opponent is not nearby"
            reward = sigmoid(PURSUER_MIN_DISTANCE_TO_OBSTACLE) - sigmoid(TRUE_DISTANCE_BETWEEN_PLAYERS)  
        else:
            state_description = "Neutral state"
            reward = 0

    # REWARD FUNCTION FOR EVADER -----------------------------------------------------------------------------------------------------------------------------------
    elif player_type == "evader":

        TRUE_DISTANCE_BETWEEN_PLAYERS = (DISTANCE_BETWEEN_PLAYERS - PURSUER_RADIUS)
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = (EVADER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE)

        if EVADER_STUCK:
            state_description = "STUCK!"
            reward = -30
        elif state["Opponent Position"] == "Tagged":
            state_description = "TAGGED!"
            reward = -30 
        # elif GAME_TIMEOUT:
        #     state_description = "Game Timeout. Evader survived!"
        #     reward = 30
        # if the other robot is nearby and there is an obstacle, there is a chance that obstacle 
        # may be the pursuer, so we discourage those states
        # or if the distance between players are very close
        elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left")  or \
               (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
               (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left") or \
               (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
               (state["Opponent Position"] == "Close Bottom")  or \
               (state["Front"] in ["Close"] and state["Opponent Position"] == "Close Front"))\
               and TRUE_DISTANCE_BETWEEN_PLAYERS <= EVADER_MIN_DISTANCE_TO_OBSTACLE)\
               or (TRUE_DISTANCE_BETWEEN_PLAYERS <= TRUE_SAFE_DISTANCE_FROM_OBSTACLE):
            
            state_description = "Pusuer is extremely close! Run away!!"
            reward = -2.5 * sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS) 

        # avoid obstacle on all sides
        elif (state["Front"] == "Close") or\
            (state["Upper Left"] in ["Close", "Too Close"]) or\
            (state["Upper Right"] in ["Close", "Too Close"]) or\
            (state["Lower Left"] in ["Close", "Too Close"]) or\
            (state["Lower Right"] in ["Close", "Too Close"]):
     
            state_description = "Obstacle nearby. Prioritize obstacle avoidance"
            # extra punishments depending on how close the pursuer is and how close the evader is to an obstacle
            reward = - sigmoid(1.0/TRUE_DISTANCE_BETWEEN_PLAYERS) - sigmoid(1.0/EVADER_MIN_DISTANCE_TO_OBSTACLE)
        
        elif state["Opponent Position"] == "Front" and TRUE_DISTANCE_BETWEEN_PLAYERS <= 1.0:
            # discourage evader from moving toward the pursuer when they are within 1.0 unit from each other
            state_description = "Pursuer is in front within 1.0 unit of distance! Go the opposite direction"
            reward = -1.0 * sigmoid(1.0/TRUE_DISTANCE_BETWEEN_PLAYERS)
        elif state["Opponent Position"] == "Front":
            state_description = "Pursuer is in front but we are not close"
            reward = -0.5 * sigmoid(1.0/TRUE_DISTANCE_BETWEEN_PLAYERS)
        elif state["Opponent Position"] == "Bottom" and TRUE_DISTANCE_BETWEEN_PLAYERS >= 0.75:
            state_description = "Pursuer is behind but not that close"
            reward = 2 * sigmoid(TRUE_DISTANCE_BETWEEN_PLAYERS) - sigmoid(PURSUER_MIN_DISTANCE_TO_OBSTACLE)
        elif state["Opponent Position"] == "Bottom" :
            state_description = "Pursuer is behind but somewhat close"
            reward = -1.0 * sigmoid(1.0/TRUE_DISTANCE_BETWEEN_PLAYERS)

        # there is no obstacle nearby and the pursuer is far away
        elif state["Upper Left"] not in ["Close", "Too Close"] and state["Lower Left"] not in ["Close", "Too Close"]\
            and state["Upper Right"] not in ["Close", "Too Close"] and state["Lower Right"] not in ["Close", "Too Close"]\
            and state["Front"] != "Close" and TRUE_DISTANCE_BETWEEN_PLAYERS >= 1.0:
            
            state_description = "Safe from obstacle and opponent is far away"
            reward = 3.5 * sigmoid(PURSUER_MIN_DISTANCE_TO_OBSTACLE) - sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS)  
        
        # there is no obstacle nearby and the pursuer is far away
        elif state["Upper Left"] not in ["Close", "Too Close"] and state["Lower Left"] not in ["Close", "Too Close"]\
            and state["Upper Right"] not in ["Close", "Too Close"] and state["Lower Right"] not in ["Close", "Too Close"]\
            and state["Front"] != "Close" and TRUE_DISTANCE_BETWEEN_PLAYERS >= SAFE_DISTANCE_FROM_OBSTACLE:
            state_description = "Safe from obstacle and opponent is within safe distance"
            reward = 2.5 * sigmoid(PURSUER_MIN_DISTANCE_TO_OBSTACLE) - sigmoid(1/TRUE_DISTANCE_BETWEEN_PLAYERS)  
        else:
            state_description = "Neutral state"
            reward = 0
    
    if verbose:
        # rospy.loginfo("DISTANCE BTW PLAYER: {}, PURSUER_MIN_DIST_OBSTACLE = {}, TRUE_SAFE_DISTANCE_FROM_OBSTACLE = {}".format(TRUE_DISTANCE_BETWEEN_PLAYERS, PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
        rospy.loginfo("{}'s state is {}".format(player_type, state))
        rospy.loginfo("{}'s state's is {}".format(player_type, state_description))
        rospy.loginfo("{}'s reward is {}".format(player_type, reward))
    
    return reward 


def get_opponent_position_rating(player_A, player_B):
    if player_A == PURSUER_POSITION:
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = PURSUER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE
    else:
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = EVADER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE
    
    
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
   
    if distance <= 0.3:
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
    if distance <= TRUE_SAFE_DISTANCE_FROM_OBSTACLE * 1.2:
        distance_rating = "Close"
    
    return (distance_rating + " " + direction_rating).strip()

def get_distance_rating(direction, distance, player_type):

    if player_type == "pursuer":
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = PURSUER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE
    else:
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = EVADER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE
    
    if direction == "Front":
        # if player_type == "evader":
        #     rospy.loginfo("TRUE_SAFE_DISTANCE_ : {}".format(TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
        #     rospy.loginfo("EVADER's front distance is : {}".format(distance))
        interval = TRUE_SAFE_DISTANCE_FROM_OBSTACLE / 1.5
        if distance <= interval * 1.35:
            rating = "Close"
        elif distance <= interval * 2.2:
            rating = "OK"
        else:
            rating = "Far"
    elif direction in ["Left", "Right", "Upper Left", "Upper Right", "Lower Left", "Lower Right"]:
        interval = TRUE_SAFE_DISTANCE_FROM_OBSTACLE/ 2.5
        if distance <= interval * 1.6:
            rating = "Too Close"
        elif distance <= (interval * 2.25):
            rating = "Close"
        elif distance <= (interval * 3.67):
            rating = "OK"
        else:
            rating = "Far"
    
    return rating


def get_current_state(message,args):
    ranges_data = message.ranges

    player_type = args["player_type"]
    verbose = args["verbose"]

    # front_sector = range(0,45) + range(315,360)
    
    # upper_left_sector = range(45,90)
    # lower_left_sector = range(90,135)
   
    # upper_right_sector = range(270,315)
    # lower_right_sector = range(225,270) 

    front_sector = range(0,30) + range(330,360)
    
    upper_left_sector = range(30,80)
    lower_left_sector = range(80,130)
   
    upper_right_sector = range(270,330)
    lower_right_sector = range(220,270) 
   
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
            "Front": get_distance_rating("Front", min_front, player_type), \
            "Upper Left" : get_distance_rating("Upper Left", min_upper_left, player_type), \
            "Upper Right": get_distance_rating("Upper Right", min_upper_right, player_type), \
            "Lower Left": get_distance_rating( "Lower Left", min_lower_left, player_type), \
            "Lower Right": get_distance_rating("Lower Right", min_lower_right, player_type), \
            "Opponent Position": get_opponent_position_rating(PURSUER_POSITION, EVADER_POSITION)
        }

        global PURSUER_MIN_DISTANCE_TO_OBSTACLE
        all_direction = ["Front", "Upper Left", "Lower Left", "Upper Right", "Lower Right"]
        all_distances =  [ min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ]
        PURSUER_MIN_DISTANCE_TO_OBSTACLE= min(all_distances)

        global PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION
        index_ = all_distances.index(PURSUER_MIN_DISTANCE_TO_OBSTACLE)
        PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION = all_direction[index_]

        if verbose:
            rospy.loginfo("Pursuer's state: {}".format(PURSUER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of pursuer's state: {}".format(reward_function("pursuer", PURSUER_STATE_DISCRETIZED, verbose=True)))
            rospy.loginfo("Min from {}".format(PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION))

    else:
        global EVADER_STATE_DISCRETIZED 
        EVADER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front, player_type), \
            "Upper Left" : get_distance_rating("Upper Left", min_upper_left, player_type), \
            "Upper Right": get_distance_rating("Upper Right", min_upper_right, player_type), \
            "Lower Left": get_distance_rating( "Lower Left", min_lower_left, player_type), \
            "Lower Right": get_distance_rating("Lower Right", min_lower_right, player_type), \
            "Opponent Position": get_opponent_position_rating(EVADER_POSITION, PURSUER_POSITION)
        }

        global EVADER_MIN_DISTANCE_TO_OBSTACLE 
        all_direction = ["Front", "Upper Left", "Lower Left", "Upper Right", "Lower Right"]
        all_distances =  [ min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ]
        EVADER_MIN_DISTANCE_TO_OBSTACLE= min(all_distances)

        global EVADER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION
        index_ = all_distances.index(EVADER_MIN_DISTANCE_TO_OBSTACLE)
        EVADER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION = all_direction[index_]

        if verbose:
            rospy.loginfo("Evader's state: {}".format(EVADER_STATE_DISCRETIZED))
            rospy.loginfo("Reward of evader's state: {}".format(reward_function("evader", EVADER_STATE_DISCRETIZED, verbose=True)))
            rospy.loginfo("Min from {}".format(EVADER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION))

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
                    # re-insert new action as new key with the q_valueOPPONENT_RATING
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
    
    # does one q-value update
    if player_type == "pursuer":
        current_state = PURSUER_STATE_DISCRETIZED 
        player_position = np.array(PURSUER_POSITION[:2])
        opponent_position = np.array(EVADER_POSITION[:2])
    else:
        current_state = EVADER_STATE_DISCRETIZED 
        player_position = np.array(EVADER_POSITION[:2])
        opponent_position = np.array(PURSUER_POSITION[:2])    
    
    rospy.loginfo("Epsilon: {}".format(epsilon))
    # get action A from S using policy
    chosen_action = get_policy(q_table, current_state, verbose = True, epsilon= epsilon)
    translation_speed, turn_action = chosen_action
    # take action A and move player, this would change the player's state
    move_robot(player_type, translation_speed, turn_action)
    # give the robot some time to apply action => proper state transition
    rospy.sleep(time_to_apply_action)
    
    # robot is now in new state S' 
    new_state = PURSUER_STATE_DISCRETIZED if (player_type == "pursuer") else EVADER_STATE_DISCRETIZED
    # robot now observes reward R(S') at this new state S'
    reward = reward_function(player_type, new_state)
    
    # update Q-value for Q(S,A)
    # Q(S,A) = Q(S,A) +  learning_rate*(reward + discount_factor* (argmax_A' Q(S', A')) - Q(S,A))
    current_state_tuple = get_state_tuple_from_dictionary(current_state)
    new_state_tuple = get_state_tuple_from_dictionary(new_state)

    best_action = get_policy(q_table, new_state, verbose=False, epsilon = 1.0)
    q_table[current_state_tuple][chosen_action] += learning_rate*(reward + \
                    discount_factor * q_table[new_state_tuple][best_action] - q_table[current_state_tuple][chosen_action])    
    
    return reward, current_state, translation_speed, turn_action


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
        translation_speed = 0.15
        # 20% chance of making random turns
        if (random.random() < random_action_chance):
            turn_angle = random.randint(0,359)
        else:
            # just go straight else
            turn_angle = 0 
    # rospy.loginfo("translation_speed: {}, turn_angle {}".format(translation_speed, turn_angle))
    return (translation_speed, turn_angle)


def manual_rescue(robot_type, time_to_apply_action = 0.5, verbose = False):
    if robot_type == "pursuer":
        robot_state = PURSUER_STATE_DISCRETIZED
    else:
        robot_state = EVADER_STATE_DISCRETIZED
    
    # don't rescue if it gets tagged before
    if robot_state["Opponent Position"] == "Tagged":
        return 

    manual_reversal(robot_type, time_to_apply_action=time_to_apply_action)
    manual_reorientation(robot_type, verbose= verbose)
    


def manual_reorientation(robot_type, time_to_apply_action=0.5, rescue_timeout_after_n_seconds = 10, verbose = False):
    
    if robot_type == "pursuer":
        robot_state = PURSUER_STATE_DISCRETIZED
    else:
        robot_state = EVADER_STATE_DISCRETIZED

    to_turn_left = (robot_state["Opponent Position"] in ["Left", "Close Left"] and robot_type == "pursuer") or\
                   (robot_state["Opponent Position"] in ["Right", "Close Right"] and robot_type == "evader")

    time_started = rospy.Time.now()
    rescue_timeout = False

    # continue trying to reorient until robot eithers find an opening in front,
    # or robot got tagged while trying to rescue itself, or the rescue timeouts (took too long)
    while not rescue_timeout and robot_state["Front"] == "Close"  and robot_state["Opponent Position"] != "Tagged":
        
        # if it takes more than 5 seconds to rescue itself, it is probably hard-stuck
        if (rospy.Time.now() - time_started) >= rospy.Duration(secs = rescue_timeout_after_n_seconds):
            rescue_timeout = True
            rescue_status = "Rescue Timeout!"
            continue 

        # if the robot could not either the robot's left side is too close to an obstacle for a valid reversal left turn 
        # or the robot's right side is too close to obstacles for a valid reversal right turn
        if (to_turn_left and (robot_state["Upper Left"] == "Too Close" and robot_state["Lower Left"] == "Too Close"))\
            or (not to_turn_left and (robot_state["Upper Right"] == "Too Close" and robot_state["Lower Right"] == "Too Close")):
            # go straight for a bit if the robot could not turn left or right
            if robot_state["Front"] != "Close":
                rescue_status = "Reorient by going straight"
                move_robot(robot_type, 0.1, 0)
            else:
                rescue_status = "Reorient by going backward"
                move_robot(robot_type, -0.1,0)

        elif to_turn_left:
        # spin robot to search for opening
            if (robot_state["Lower Right"] == "Too Close" and robot_state["Upper Right"] == "Too Close"):
                move_robot(robot_type, -0.2, -40)
                rescue_status = "Reorient to face left by reversing to left side"
            else:
                move_robot(robot_type,0, 40)
                # move_robot(robot_type, 0, 20)
                rescue_status = "Reorient to face left by spinning to right side"
        else:
            if (robot_state["Lower Left"] == "Too Close" and robot_state["Upper Left"] == "Too Close"):
                move_robot(robot_type, -0.2, 40)
                rescue_status = "Reorient to face right by reversing to right side"
            else:
                move_robot(robot_type,0, -40)
                rescue_status = "Reorient to face right by spinning to left side"

        rospy.sleep(time_to_apply_action)
        
        # fetch new robot state
        if robot_type == "pursuer":
            robot_state = PURSUER_STATE_DISCRETIZED
        else:
            robot_state = EVADER_STATE_DISCRETIZED
        
        if verbose:
            rospy.loginfo(rescue_status)

        # if the robot gets tagged while rescuing itself, stop the rescue
        if robot_state["Opponent Position"] == "Tagged":
            return 

    if rescue_timeout:
        if robot_type == "pursuer":
            global RESCUE_PURSUER_FAILED
            global PURSUER_STUCK

            RESCUE_PURSUER_FAILED = True
            PURSUER_STUCK = True
        else:
            global RESCUE_EVADER_FAILED
            global EVADER_STUCK

            RESCUE_EVADER_FAILED = True
            EVADER_STUCK = True
    
    # stop robot once opponent is in front
    move_robot(robot_type, 0, 0)
    
    

def manual_reversal(robot_type, time_to_apply_action=1.5):
    if robot_type == "pursuer":
        robot_state = PURSUER_STATE_DISCRETIZED
    else:
        robot_state = EVADER_STATE_DISCRETIZED

    if robot_state["Opponent Position"] == "Tagged":
        return 

    close_ratings = ["Too Close", "Close"]
    
    translation_speed = -0.15
    # turn_angle = 0
    
    if (robot_state["Upper Left"] in close_ratings or robot_state["Lower Left"] in close_ratings)\
        and (robot_state["Upper Right"] in close_ratings or robot_state["Lower Right"] in close_ratings):
        # just reverse backward
        # rospy.loginfo("Reversing backward")
        turn_angle = 0
    elif robot_type == "pursuer" and robot_state["Opponent Position"] in ["Left", "Close Left"]:
        # rospy.loginfo("Right turn while reversing")    
        # right turn while reversing so pursuer could face the evader
        turn_angle = 60
    elif robot_type == "pursuer" and robot_state["Opponent Position"] in ["Right", "Close Right"]:
        # left turn while reversing so pursuer could face the evader to the right
        # rospy.loginfo("Right turn while reversing")   
        turn_angle = -60
    elif robot_type == "evader" and robot_state["Opponent Position"] in ["Left", "Close Left"]:
        # left turn while reversing so evader could face AWAY from the pursuer to the left
        turn_angle = -60
    else:
        turn_angle = 60
    
    move_robot(robot_type, translation_speed, turn_angle)
    
    rospy.sleep(time_to_apply_action)
    # rospy.loginfo("Slept")

def follow_policy(player_type, q_table, time_to_apply_action = 0.33, evader_random_walk = False):
    
    if player_type == "pursuer":
        current_state = PURSUER_STATE_DISCRETIZED
        translation_velocity, angular_velocity = get_policy(q_table, current_state, verbose= False, epsilon = 1.0)

    else:
        current_state = EVADER_STATE_DISCRETIZED
        if evader_random_walk:
            translation_velocity, angular_velocity = random_walk_behavior(robot_type="evader", robot_state=current_state)
        else:
            translation_velocity, angular_velocity = get_policy(q_table, current_state, verbose= False, epsilon = 1.0)
       
    move_robot(player_type, translation_velocity, angular_velocity)
    rospy.sleep(time_to_apply_action)

def is_stuck(last_few_positions, robot_state):
    # Checking if the robot is stuck requires info about 
    # whether it is near an obstacle and if its location has not changed in a while.
     
    # Checking if the location hasn't changed alone is not sufficient 
    # since the robot could be moving very slowly => the algorithm thinks it is stuck
    is_stuck = False
    if len(last_few_positions) > 0 and last_few_positions is not None:
        changes_in_x = 0
        changes_in_y = 0
        for i in range(1,len(last_few_positions)):
            changes_in_x += abs(last_few_positions[i][0] - last_few_positions[i - 1][0])
            changes_in_y += abs(last_few_positions[i][1] - last_few_positions[i - 1][1])
        
        # if accumulated changes in both coordinates are less than a very small number, 
        # the robot is probably stuck
        is_in_same_place = changes_in_x < 0.05 and changes_in_y < 0.05

        # only check if robot's front is stuck, since if its side is stuck, it could rescue itself by turning the opposite direction
        is_near_obstacle = robot_state["Front"] == "Close" 

        # the robot is consider stuck of it is near an obstacle and hasn't changed position in a while
        is_stuck = is_near_obstacle and is_in_same_place
    return is_stuck

def is_terminal_state(train_type, game_timeout, pursuer_stuck, evader_stuck, opponent_rating, verbose=True):
    if opponent_rating == "Tagged": 
        is_terminal = True
        terminal_status = "Terminated because TAGGED. Pursuer Won"
    elif game_timeout:
        is_terminal = True
        terminal_status = "Terminated because game-timeot. Evader won"
    # if we are just training the pursuer, even if the evader gets stuck
    # we still let the pursuer run until it catches the evader, or gets stucks itself
    elif train_type == "pursuer" and pursuer_stuck:
        is_terminal = True
        terminal_status = "Terminated because pursuer is STUCK"
    # when training the evader, terminate when the evader gets stuck
    elif train_type == "evader" and evader_stuck:
        is_terminal = True
        terminal_status = "Terminated because evader is STUCK"
    else:
        is_terminal = False 
    

    if is_terminal and verbose:
        rospy.loginfo(terminal_status)
    
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

def train(train_type = "pursuer", total_episodes = 1000, learning_rate = 0.2, discount_factor = 0.8, starting_epsilon = 0.2, max_epsilon = 0.9, episode_time_limit = 30, time_to_apply_action=0.5, evader_random_walk = False, do_initial_test = False):
    
    if train_type not in ["pursuer", "evader"]:
        rospy.loginfo("Unrecognized train type. Either \"puruser\" or \"evader\"")
        return 
    
    current_episode = 0
    # accumulated_pursuer_reward = 0
    # accumulated_evader_reward = 0
    epsilon = starting_epsilon

    # create subplot of 2 rows, 3 columns
    fig, ax = plt.subplots(2,3)

    if train_type == "pursuer":
        title = "Pursuer Training Progress Dashboard"
    elif train_type == "evader":
        title = "Evader Training Progress Dashboard"
    else:
        title = "Both Evader and Pursuer Training Progress Dashboard"
    
    # figure's super title   
    fig.suptitle(title, fontsize=16)
    # plot learning curve as robot learns
    learning_curve, = ax[0,0].plot([],[], "r-", label="Q-learning TD")
    test_curve, = ax[0,0].plot([],[], linestyle="-", marker="x", color="k", label="Q-learning TD Test-Phase Reward")
    ax[0,0].set_xlabel("Training episode")
    ax[0,0].set_ylabel("Accumulated rewards")
    ax[0,0].set_xlim(0 , total_episodes)
    ax[0,0].set_ylim(-100, 100)
    ax[0,0].set_title("Accumulated Rewards vs Training episodes")
    ax[0,0].legend(loc="upper left")
    ax[0,0].axhline(y= 0, color = "g", linestyle = "-")
    
    # subplot on row 0 column 1 shows details regarding how the robot is doing each test phase
    tag_curve, = ax[0,1].plot([],[], "g-", marker="x", label="Number of tags in test phase")
    stuck_curve, = ax[0,1].plot([],[], "r-", marker="x", label="Number of times stuck in test phase")
    timeout_curve, = ax[0,1].plot([],[], "b-",  marker="x", label="Number of timeouts in test phase")
    ax[0,1].set_xlabel("Number of Episodes")
    ax[0,1].set_ylabel("Number of scenarios in Test Phase")
    ax[0,1].set_xlim(0, total_episodes)
    ax[0,1].set_ylim(0, 40)
    ax[0,1].yaxis.set_ticks(np.arange(0, 41, 1))
    ax[0,1].set_title("Test Phase Details")
    ax[0,1].legend(loc="upper left")


    # go_left_when_opponent_bottom_curve, = ax[0,2].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
    # go_right_when_opponent_bottom_curve, = ax[0,2].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
    # go_front_when_opponent_bottom_curve, = ax[0,2].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
    # ax[0,2].set_xlabel("Number of Episodes")
    # ax[0,2].set_ylabel("Proportion of actions chosen")
    # ax[0,2].set_xlim(0, total_episodes)
    # ax[0,2].set_ylim(0, 1.0)
    # ax[0,2].legend(loc="upper left")
    # ax[0,2].set_title("Proportion of actions chosen when opponent is BEHIND")

    if train_type == "pursuer":
        average_distance_at_terminal_curve, = ax[0,2].plot([],[], "g-", marker="x", label = "Average distance at terminal state")
        ax[0,2].set_xlabel("Number of Episodes")
        ax[0,2].set_ylabel("Average distance between players")
        ax[0,2].set_xlim(0, total_episodes)
        ax[0,2].set_ylim(0, 6.0)
        ax[0,2].legend(loc="upper right")
        ax[0,2].set_title("Average distance between players after game ends")
    else:
        average_time_at_terminal_curve, = ax[0,2].plot([],[], "g-", marker="x", label = "Average time survived")
        ax[0,2].set_xlabel("Number of Episodes")
        ax[0,2].set_ylabel("Average time suvived")
        ax[0,2].set_xlim(0, total_episodes)
        ax[0,2].set_ylim(0, episode_time_limit)
        ax[0,2].legend(loc="upper right")
        ax[0,2].set_title("Average time survived by evader")


    go_left_when_opponent_left_curve, = ax[1,0].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
    go_right_when_opponent_left_curve, = ax[1,0].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
    go_front_when_opponent_left_curve, = ax[1,0].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
    ax[1,0].set_xlabel("Number of Episodes")
    ax[1,0].set_ylabel("Proportion of actions chosen")
    ax[1,0].set_xlim(0, total_episodes)
    ax[1,0].set_ylim(0, 1.0)
    ax[1,0].legend(loc="upper left")
    ax[1,0].set_title("Proportion of actions chosen when opponent is to the LEFT")

    go_left_when_opponent_right_curve, = ax[1,1].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
    go_right_when_opponent_right_curve, = ax[1,1].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
    go_front_when_opponent_right_curve, = ax[1,1].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
    ax[1,1].set_xlabel("Number of Episodes")
    ax[1,1].set_ylabel("Proportion of actions chosen")
    ax[1,1].set_xlim(0, total_episodes)
    ax[1,1].set_ylim(0, 1.0)
    ax[1,1].legend(loc="upper left")
    ax[1,1].set_title("Proportion of actions chosen when opponent is to the RIGHT")


    go_left_when_opponent_front_curve, = ax[1,2].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
    go_right_when_opponent_front_curve, = ax[1,2].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
    go_front_when_opponent_front_curve, = ax[1,2].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
    ax[1,2].set_xlabel("Number of Episodes")
    ax[1,2].set_ylabel("Proportion of actions chosen")
    ax[1,2].set_xlim(0, total_episodes)
    ax[1,2].set_ylim(0, 1.0)
    ax[1,2].legend(loc="upper left")
    ax[1,2].set_title("Proportion of actions chosen when opponent is IN FRONT")
 
    plt.show(block=False)

    best_test_score = float("-inf")
    best_train_score = float("-inf")
    training_reward = 0

    # check state-action convergence
    num_times_go_left_opponent_is_left = 0
    num_times_go_right_opponent_is_left = 0
    num_times_go_front_opponent_is_left = 0

    num_times_go_left_opponent_is_right = 0
    num_times_go_right_opponent_is_right = 0
    num_times_go_front_opponent_is_right = 0

    num_times_go_left_opponent_is_front = 0
    num_times_go_right_opponent_is_front = 0
    num_times_go_front_opponent_is_front = 0

    # num_times_go_left_opponent_is_bottom = 0
    # num_times_go_right_opponent_is_bottom = 0
    # num_times_go_front_opponent_is_bottom = 0
    accumulated_distance_between_players_at_end = 0.0
    accumulated_time_survived_by_evader = rospy.Duration(secs = 0)
    
    while current_episode < total_episodes:
        if (PURSUER_STATE_DISCRETIZED is not None and EVADER_STATE_DISCRETIZED is not None):
            rospy.loginfo("Starting Episode {}".format(current_episode))
            print("*"*50)
 
            if train_type == "pursuer":
                player = "pursuer"
                opponent = "evader"
                q_table_player = Q_TABLE_PURSUER
                q_table_opponent = Q_TABLE_EVADER
            else:
                player = "evader"
                opponent = "pursuer"
                q_table_player = Q_TABLE_EVADER
                q_table_opponent = Q_TABLE_PURSUER


            rospy.loginfo("Player being trained {}".format(player))
            # keep track of whether pursuer and evader are stuck, and what time
            global EVADER_STUCK
            global PURSUER_STUCK

            PURSUER_STUCK = False
            EVADER_STUCK = False

            global RESCUE_EVADER_FAILED
            global RESCUE_PURSUER_FAILED
            RESCUE_EVADER_FAILED = False
            RESCUE_PURSUER_FAILED = False

            last_few_pursuer_positions = []
            last_few_evader_positions = []

            # spawn robots at semi-random locations
            spawn_robots()

            # every 400 episodes, test the policy learned so far 
            # do an intial test at epsidode 0 if specified
            if (current_episode == 0 and do_initial_test) or (current_episode % 400 == 0 and current_episode != 0):
                rospy.loginfo("Testing policy learned so far")
                if train_type == "pursuer":
                    test_reward, num_tagged, num_stuck, num_timeout = test(player, total_episodes = 40, episode_time_limit=episode_time_limit, time_to_apply_action=time_to_apply_action, evader_random_walk= evader_random_walk, allow_evader_manual_rescue= True)
                elif train_type == "evader":
                    # rospy.loginfo(player_to_train)
                    test_reward, num_tagged, num_stuck, num_timeout = test(player, total_episodes = 40, episode_time_limit=episode_time_limit, time_to_apply_action=time_to_apply_action, evader_random_walk= evader_random_walk, allow_pursuer_manual_rescue= True)
                if test_reward > best_test_score:
                    # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                    # save q-table
                    with open("q_table_{}_best_testing.txt".format(player), "w") as q_table_file:
                        q_table_file.seek(0)
                        q_table_file.write(pickle.dumps(q_table_player)) 
                    best_test_score = test_reward
                # plot learning cureve with test rewards
                _plot_learning_curve(test_curve, current_episode, test_reward)
                # plot tag curve, stuck curve and timeout curve
                _plot_learning_curve(tag_curve, current_episode, num_tagged)
                _plot_learning_curve(stuck_curve, current_episode, num_stuck)
                _plot_learning_curve(timeout_curve, current_episode, num_timeout)
            
    
            # every <epsilon_update_interval> training episodes, the epsilon goes up by 0.05 to encourage more exploitation and less exploration
            # as the robot learns more and more about the environment
            epsilon_update_interval = int(total_episodes / (((max_epsilon - starting_epsilon) / 0.05) + 1))
            if current_episode != 0 and current_episode % epsilon_update_interval == 0 and epsilon < max_epsilon:
                epsilon += 0.05
                # plot training episode where epsilon changes
                # ax[0,0].axvline(x=current_episode, color="g",linestyle="--" )
                # ax[0,0].annotate("Epsilon: {}".format(epsilon),(current_episode, 2.5 *  (current_episode/epsilon_update_interval)))
            
            accumulated_reward = 0
            
            global GAME_TIMEOUT
            GAME_TIMEOUT = False

            start_time = rospy.Time.now()
            time_elapsed = rospy.Duration(secs=0)
            time_spent_on_manual_rescue = rospy.Duration(secs=0)
            
            while(not is_terminal_state(train_type, GAME_TIMEOUT, PURSUER_STUCK, EVADER_STUCK, PURSUER_STATE_DISCRETIZED["Opponent Position"])):
                # get time elapsed so far 
                time_elapsed = (rospy.Time.now() - start_time) - time_spent_on_manual_rescue
                
                # check if the game has timeout
                if time_elapsed >= rospy.Duration(secs = episode_time_limit):
                    GAME_TIMEOUT = True
                
                # initialize a seperate thread to handle rescuing the opponent should it gets stuck
                # since we are training our player, we assume the opponent is already good
                # training episode would only terminate if the player gets stuck but not when the opponent gets stuck 
                rescue_thread = threading.Thread(target=manual_rescue, args = (opponent, 1.0))
                
                # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 1.5 seconds
                if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                    PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=PURSUER_STATE_DISCRETIZED)
                    if PURSUER_STUCK and not GAME_TIMEOUT:
                        if train_type == "evader": 
                            # if we are training the evader, the pursuer could manually reverse to rescue itself when stuck
                            # and resume chasing the evader
                            # rescue_start_time = rospy.Time.now()
                            rescue_thread.start()
                            last_few_pursuer_positions = []
                            # get new state after reversal
                            PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=PURSUER_STATE_DISCRETIZED)
                    if len(last_few_pursuer_positions) != 0:
                        del last_few_pursuer_positions[0]
                    
                if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                    EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=EVADER_STATE_DISCRETIZED)
                    if EVADER_STUCK and not GAME_TIMEOUT:
                        if train_type == "pursuer" and not evader_random_walk: 
                            # when training the pursuer, we have two modes
                            # we could either train it against a random-walking evader that has a reversal already coded in
                            # or we could train it against an evader which uses a q-table with no reversal actions.
                            # when we are training against the latter, we have to call the manual rescue thread should
                            # the evader gets stuck since it does not have the reversal action in its q-table
                            # rescue_start_time = rospy.Time.now()
                            rescue_thread.start()
                            last_few_evader_positions = []
                            # get new state after reversal
                            EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=EVADER_STATE_DISCRETIZED)
                    if len(last_few_evader_positions) != 0:
                        del last_few_evader_positions[0]

                last_few_pursuer_positions.append(PURSUER_POSITION[:2])
                last_few_evader_positions.append(EVADER_POSITION[:2])
                

                while (rescue_thread.is_alive() and not evader_random_walk):
                    # when waiting for the other robot to rescue itself, the current robot continue learning
                    # move_robot(player, 0,0)
                    q_learning_td(player, q_table_player, learning_rate = learning_rate, discount_factor = discount_factor, epsilon = epsilon,\
                    time_to_apply_action = time_to_apply_action)
                    rescue_thread.join()
                    # rescue_stop_time = rospy.Time.now()
                    # time_spent_on_manual_rescue += (rescue_stop_time - rescue_start_time)

                if not GAME_TIMEOUT and (RESCUE_PURSUER_FAILED or RESCUE_EVADER_FAILED):
                    # if failed to rescue, break out and restart episode
                    if RESCUE_EVADER_FAILED and RESCUE_PURSUER_FAILED:
                        rospy.loginfo("RESCUE FAILED FOR BOTH")
                    elif RESCUE_PURSUER_FAILED:
                        rospy.loginfo("RESCUE PURSUER FAILED")
                    else:
                        rospy.loginfo("RESCUE EVADER FAILED")
                    break


                # run opponent's decision-making in seperate thread
                if train_type == "pursuer" and evader_random_walk:
                    opponent_decision_making_thread = threading.Thread(target = follow_policy, args=(opponent, q_table_opponent, time_to_apply_action, True))
                    opponent_decision_making_thread.start()
                else:
                    opponent_decision_making_thread = threading.Thread(target = follow_policy, args=(opponent, q_table_opponent, time_to_apply_action, False))
                    opponent_decision_making_thread.start()

                # have current robot train using q-learning
                reward, current_state, translation_speed, turn_action = q_learning_td(player, q_table_player, learning_rate = learning_rate, discount_factor = discount_factor, epsilon = epsilon,\
                    time_to_apply_action = time_to_apply_action)
                
                # accumulate rewards
                accumulated_reward += reward

                # accumulate number of times each scenarios happen to check for state-action convergence
                if "Left" in current_state["Opponent Position"]:
                    if turn_action in [60, 40, 20]:
                        num_times_go_left_opponent_is_left += 1
                    elif turn_action in [-60, -40, -20]:
                        num_times_go_right_opponent_is_left += 1
                    else:
                        num_times_go_front_opponent_is_left += 1
                elif "Right" in current_state["Opponent Position"]:
                    if turn_action in [60, 40, 20]:
                        num_times_go_left_opponent_is_right += 1
                    elif turn_action in [-60, -40, -20]:
                        num_times_go_right_opponent_is_right += 1
                    else:
                        num_times_go_front_opponent_is_right += 1
                elif "Front" in current_state["Opponent Position"]:
                    if turn_action in [60, 40, 20]:
                        num_times_go_left_opponent_is_front += 1
                    elif turn_action in [-60, -40, -20]:
                        num_times_go_right_opponent_is_front += 1
                    else:
                        num_times_go_front_opponent_is_front += 1
                # else:
                #     if turn_action in [60, 40, 20]:
                #         num_times_go_left_opponent_is_bottom += 1
                #     elif turn_action in [-60, -40, -20]:
                #         num_times_go_right_opponent_is_bottom += 1
                #     else:
                #         num_times_go_front_opponent_is_bottom += 1   
            
                # follow_policy(opponent_to_test, q_table_opponent)
                opponent_decision_making_thread.join()
            
            # restart episode if manual rescue failed, don't count
            if not RESCUE_EVADER_FAILED and not RESCUE_PURSUER_FAILED:
                current_episode += 1

            # accumulate metrics
            if DISTANCE_BETWEEN_PLAYERS != None:
                accumulated_distance_between_players_at_end += DISTANCE_BETWEEN_PLAYERS
            # accumulate time the evader survived on the map so far
            accumulated_time_survived_by_evader += time_elapsed
            training_reward += accumulated_reward
            

            if current_episode % 250 == 0:
                # plot learning curve using the average reward each 250 training episodes
                _plot_learning_curve(learning_curve,current_episode, training_reward/250)
                if training_reward > best_train_score:
                    # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                    # save q-table
                    with open("q_table_{}_best_training.txt".format(player), "w") as q_table_file:
                        q_table_file.seek(0)
                        q_table_file.write(pickle.dumps(q_table_player)) 
                    best_train_score = training_reward
                training_reward = 0

                # plot the average distance at terminal state for every 250 episodes
                if train_type == "pursuer":
                    _plot_learning_curve(average_distance_at_terminal_curve, current_episode, accumulated_distance_between_players_at_end/250.0)
                    # reset
                    accumulated_distance_between_players_at_end = 0
                else:
                    _plot_learning_curve(average_time_at_terminal_curve, current_episode, (accumulated_time_survived_by_evader/250.0).to_sec())
                    accumulated_time_survived_by_evader = rospy.Duration(secs = 0)

            if current_episode % 500 == 0:
                # plot state-action convergence graph
                # when proportion of action when opponent is to the left
                _plot_learning_curve(go_left_when_opponent_left_curve, current_episode, float(num_times_go_left_opponent_is_left)/sum([num_times_go_front_opponent_is_left, num_times_go_left_opponent_is_left, num_times_go_right_opponent_is_left]))
                _plot_learning_curve(go_right_when_opponent_left_curve, current_episode, float(num_times_go_right_opponent_is_left)/sum([num_times_go_front_opponent_is_left, num_times_go_left_opponent_is_left, num_times_go_right_opponent_is_left]))
                _plot_learning_curve(go_front_when_opponent_left_curve, current_episode, float(num_times_go_front_opponent_is_left)/sum([num_times_go_front_opponent_is_left, num_times_go_left_opponent_is_left, num_times_go_right_opponent_is_left]))
                # when proportion of action when opponent is to the right
                _plot_learning_curve(go_left_when_opponent_right_curve, current_episode, float(num_times_go_left_opponent_is_right)/sum([num_times_go_front_opponent_is_right, num_times_go_left_opponent_is_right, num_times_go_right_opponent_is_right]))
                _plot_learning_curve(go_right_when_opponent_right_curve, current_episode, float(num_times_go_right_opponent_is_right)/sum([num_times_go_front_opponent_is_right, num_times_go_left_opponent_is_right, num_times_go_right_opponent_is_right]))
                _plot_learning_curve(go_front_when_opponent_right_curve, current_episode, float(num_times_go_front_opponent_is_right)/sum([num_times_go_front_opponent_is_right, num_times_go_left_opponent_is_right, num_times_go_right_opponent_is_right]))
                # when proportion of action when opponent is in front
                _plot_learning_curve(go_left_when_opponent_front_curve, current_episode, float(num_times_go_left_opponent_is_front)/sum([num_times_go_front_opponent_is_front, num_times_go_left_opponent_is_front, num_times_go_right_opponent_is_front]))
                _plot_learning_curve(go_right_when_opponent_front_curve, current_episode, float(num_times_go_right_opponent_is_front)/sum([num_times_go_front_opponent_is_front, num_times_go_left_opponent_is_front, num_times_go_right_opponent_is_front]))
                _plot_learning_curve(go_front_when_opponent_front_curve, current_episode, float(num_times_go_front_opponent_is_front)/sum([num_times_go_front_opponent_is_front, num_times_go_left_opponent_is_front, num_times_go_right_opponent_is_front]))
                
                # # when proportion of action when opponent is behind
                # _plot_learning_curve(go_left_when_opponent_bottom_curve, current_episode, float(num_times_go_left_opponent_is_bottom)/sum([num_times_go_front_opponent_is_bottom, num_times_go_left_opponent_is_bottom, num_times_go_right_opponent_is_bottom]))
                # _plot_learning_curve(go_right_when_opponent_bottom_curve, current_episode, float(num_times_go_right_opponent_is_bottom)/sum([num_times_go_front_opponent_is_bottom, num_times_go_left_opponent_is_bottom, num_times_go_right_opponent_is_bottom]))
                # _plot_learning_curve(go_front_when_opponent_bottom_curve, current_episode, float(num_times_go_front_opponent_is_bottom)/sum([num_times_go_front_opponent_is_bottom, num_times_go_left_opponent_is_bottom, num_times_go_right_opponent_is_bottom]))
                
                # reset metrics
                num_times_go_left_opponent_is_left = 0
                num_times_go_right_opponent_is_left = 0
                num_times_go_front_opponent_is_left = 0

                num_times_go_left_opponent_is_right = 0
                num_times_go_right_opponent_is_right = 0
                num_times_go_front_opponent_is_right = 0

                num_times_go_left_opponent_is_front = 0
                num_times_go_right_opponent_is_front = 0
                num_times_go_front_opponent_is_front = 0

                # num_times_go_left_opponent_is_bottom = 0
                # num_times_go_right_opponent_is_bottom = 0
                # num_times_go_front_opponent_is_bottom = 0
                
            if current_episode != 0 and current_episode % 500 == 0:
                # save the training curve figure every 500 episodes
                plt.savefig("td_curve_{}_{}_episodes".format(train_type, current_episode), dpi=100)
            
            # save q-table every training episode
            with open("q_table_{}.txt".format(player), "w") as q_table_file:
                q_table_file.seek(0)
                q_table_file.write(pickle.dumps(q_table_player)) 
            # notify user that Q-table has been saved
            rospy.loginfo("Saved Q-Table_{}".format(player))


def test(player, total_episodes = 2, episode_time_limit=30, time_to_apply_action = 0.5, allow_pursuer_manual_rescue= False, allow_evader_manual_rescue = False, evader_random_walk = False):
    current_episode = 0
    current_state = None
    accumulated_reward = 0 
    # keeps track of how many rounds the pursuer managed to tag the evader
    num_tagged = 0
    num_pursuer_stuck = 0
    num_evader_stuck = 0
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

        global RESCUE_EVADER_FAILED
        global RESCUE_PURSUER_FAILED
        RESCUE_EVADER_FAILED = False
        RESCUE_PURSUER_FAILED = False
        
        # spawn at random points
        spawn_robots()

        # fetch new state when robots respawn for a new testing episode
        while current_state is None or current_state["Opponent Position"] == "Tagged":
            if player == "pursuer":
                current_state = PURSUER_STATE_DISCRETIZED
                opponent = "evader"
                q_table_player = Q_TABLE_PURSUER
                q_table_opponent = Q_TABLE_EVADER
            else:
                current_state = EVADER_STATE_DISCRETIZED
                opponent = "pursuer"
                q_table_player = Q_TABLE_EVADER
                q_table_opponent = Q_TABLE_PURSUER
        
        # keeps track of how much time is left in current round
        start_time = rospy.Time.now()
        # time_elapsed = rospy.Duration(secs=0)
        
        global GAME_TIMEOUT
        GAME_TIMEOUT = False
        
        # keep track of total time spent on rescuing the robot
        # rescue time will not count toward game time
        time_spent_on_manual_rescue = rospy.Duration(secs=0)
        while(not is_terminal_state(train_type=player, game_timeout= GAME_TIMEOUT ,pursuer_stuck= PURSUER_STUCK, evader_stuck= EVADER_STUCK, \
                                    opponent_rating = current_state["Opponent Position"], verbose=True)):
            
            # initialize the rescue-reversal behavior for the opponent since we want to 
            # test how our agent performs around an adversarial opponent
            # initialize two rescue threads, they might be used or not at all depending on the parameters 
            # allow_pursuer_manual_rescue or allow_evader_manual_rescue function call
            evader_rescue_thread = threading.Thread(target=manual_rescue, args = ("evader", 1.0))
            pursuer_rescue_thread = threading.Thread(target=manual_rescue, args = ("pursuer", 1.0))

            # don't count time to manually rescue players as part of game time
            time_elapsed = (rospy.Time.now() - start_time) - time_spent_on_manual_rescue
            
            # check if the game has timeout
            if time_elapsed >= rospy.Duration(secs = episode_time_limit):
                GAME_TIMEOUT = True
                num_timeout += 1

            # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 1.50 seconds
            if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=PURSUER_STATE_DISCRETIZED)
                if PURSUER_STUCK and not GAME_TIMEOUT:
                    num_pursuer_stuck += 1
                    # starts rescue thread if we are either
                    # testing how the robot performs with manual rescue on
                    # or if we are testing the evader against an good, adversarial pursuer
                    if (player == "pursuer" and allow_pursuer_manual_rescue) or (player == "evader"):
                        pursuer_rescue_start_time = rospy.Time.now()
                        pursuer_rescue_thread.start()
                        last_few_pursuer_positions = []
                        # get new state after reversal
                        PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=PURSUER_STATE_DISCRETIZED)
                if len(last_few_pursuer_positions) != 0:
                    del last_few_pursuer_positions[0]
                
            if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=EVADER_STATE_DISCRETIZED)
                if EVADER_STUCK and not GAME_TIMEOUT:
                    num_evader_stuck += 1
                    if (player == "pursuer" and not evader_random_walk) or (player == "evader" and allow_evader_manual_rescue): 
                        # when training the pursuer, we have two modes
                        # we could either train it against a random-walking evader that has a reversal already coded in
                        # or we could train it against an evader which uses a q-table with no reversal actions.
                        # when we are training against the latter, we have to call the manual rescue thread should
                        # the evader gets stuck since it does not have the reversal action in its q-table
                        evader_rescue_start_time = rospy.Time.now()
                        evader_rescue_thread.start()
                        last_few_evader_positions = []
                        # get new state after reversal
                        EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=EVADER_STATE_DISCRETIZED)
                if len(last_few_evader_positions) != 0:
                    # rospy.loginfo(EVADER_STUCK)
                    del last_few_evader_positions[0]

            last_few_pursuer_positions.append(PURSUER_POSITION[:2])
            last_few_evader_positions.append(EVADER_POSITION[:2])

            # wait for the rescue threads to join
            while pursuer_rescue_thread.is_alive():
                # while waiting for pursuer to unstuck itself, continue moving the evader
                follow_policy(player_type= "evader", q_table= Q_TABLE_EVADER, time_to_apply_action=time_to_apply_action)
                # move_robot("evader", 0,0)
                pursuer_rescue_thread.join()
                # pursuer_rescue_stop_time = rospy.Time.now()
                # time_spent_on_manual_rescue += (pursuer_rescue_stop_time - pursuer_rescue_start_time)
            

            while evader_rescue_thread.is_alive():
               # while waiting for evader to unstuck itself, continue moving the pursuer
                follow_policy(player_type= "pursuer", q_table= Q_TABLE_PURSUER, time_to_apply_action=time_to_apply_action)
                evader_rescue_thread.join()
                # evader_rescue_stop_time = rospy.Time.now()
                # time_spent_on_manual_rescue += (evader_rescue_stop_time - evader_rescue_start_time)

            # check if rescue threads were successful at rescuing the robots from being stuck
            # if not break out of current episode and restart
            if not GAME_TIMEOUT and (RESCUE_PURSUER_FAILED or RESCUE_EVADER_FAILED):
                # if failed to rescue, break out and restart episode
                if RESCUE_EVADER_FAILED and RESCUE_PURSUER_FAILED:
                    rospy.loginfo("RESCUE FAILED FOR BOTH")
                elif RESCUE_PURSUER_FAILED:
                    rospy.loginfo("RESCUE PURSUER FAILED")
                else:
                    rospy.loginfo("RESCUE EVADER FAILED")
                break
            
            # if everything is good, then we proceed to move robots as usual

            # run opponent's decision-making in seperate thread
            if player == "pursuer" and evader_random_walk:
                # if we are testing the pursuer against a random-walking evader
                opponent_decision_making_thread = threading.Thread(target = follow_policy, args=(opponent, q_table_opponent, time_to_apply_action, True))
                opponent_decision_making_thread.start()
            else:
                # else if we are testing the evader or testing the pursuer against an adversarial evader that uses its own q-table
                opponent_decision_making_thread = threading.Thread(target = follow_policy, args=(opponent, q_table_opponent, time_to_apply_action))
                opponent_decision_making_thread.start()
        
            # player's decision making
            follow_policy(player_type=player,q_table= q_table_player, time_to_apply_action=time_to_apply_action)

            # wait for opponent thread to finish before moving on
            opponent_decision_making_thread.join()
            
        
            # observe rewards of new state
            if player == "pursuer":
                current_state = PURSUER_STATE_DISCRETIZED
            else:
                current_state = EVADER_STATE_DISCRETIZED
            
            accumulated_reward += reward_function(player, current_state, verbose=False)

            if not GAME_TIMEOUT and current_state["Opponent Position"] == "Tagged":
                num_tagged += 1

        if not RESCUE_EVADER_FAILED and not RESCUE_PURSUER_FAILED:    
            current_episode += 1

    # print diagnostic test phase details   
    rospy.loginfo("\nTEST PHASE DETAIL\n")    
    rospy.loginfo("TEST PHASE: {} times pursuer tagged the evader out of {} testing rounds".format(num_tagged, total_episodes))
    if player == "pursuer" and allow_pursuer_manual_rescue:
        rospy.loginfo("TEST PHASE: {} times the pursuer have to use the manual rescuing function to unstuck itself out of {} testing rounds".format(num_pursuer_stuck, total_episodes))
    elif player == "pursuer":
        rospy.loginfo("TEST PHASE: {} times the pursuer got stuck out of {} testing rounds".format(num_pursuer_stuck, total_episodes))
    elif player == "evader" and allow_pursuer_manual_rescue:
        rospy.loginfo("TEST PHASE: {} times the evader have to use the manual rescuing function to unstuck itself out of {} testing rounds".format(num_evader_stuck, total_episodes))
    else:
        rospy.loginfo("TEST PHASE: {} times the evader got stuck out of {} testing rounds".format(num_evader_stuck, total_episodes))
    rospy.loginfo("TEST PHASE: {} times the evader survived the match against the pursuer (timeout) out of {} testing rounds".format(num_timeout, total_episodes))
    rospy.loginfo("-"*50)
    
    if player == "pursuer":
        num_stuck = num_pursuer_stuck
    else:
        num_stuck = num_evader_stuck
    return (accumulated_reward / total_episodes, num_tagged, num_stuck, num_timeout)

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


    # load_q_table(q_table_name="q_table_evader_best_training.txt", player_type="evader")
    # train(train_type = "pursuer", starting_epsilon=0.4, max_epsilon=0.95, total_episodes=25000, episode_time_limit=45, time_to_apply_action=0.5, evader_random_walk=False, do_initial_test=False)

    load_q_table(q_table_name="q_table_pursuer_best_training.txt", player_type="pursuer")
    train(train_type = "evader", starting_epsilon=0.45, max_epsilon=0.95, total_episodes=30000, episode_time_limit=45, time_to_apply_action=0.5, evader_random_walk=False, do_initial_test=False)
    
    # rospy.loginfo("Result from PURSUER BEST TRAINING")
    # successfully_loaded = load_q_table(q_table_name="q_table_pursuer_best_training.txt", player_type="pursuer")
    # if successfully_loaded:
    #     test("pursuer", total_episodes= 50, episode_time_limit=90, allow_pursuer_manual_rescue=True, time_to_apply_action= 0.5, evader_random_walk= False)

    rospy.loginfo("Result from EVADER BEST TRAINING")
    successfully_loaded = load_q_table(q_table_name="q_table_evader_best_training.txt", player_type="evader")
    if successfully_loaded:
        test("evader", total_episodes= 50, episode_time_limit=90, allow_evader_manual_rescue=True, time_to_apply_action= 0.5, evader_random_walk= False)

    # rospy.loginfo("Result from BEST TESTING")
    # successfully_loaded = load_q_table(q_table_name="q_table_evader_best_testing.txt", player_type="evader")
    # if successfully_loaded:
    #     test("evader", total_episodes= 100, episode_time_limit=90, allow_evader_manual_rescue=True, time_to_apply_action= 0.5, evader_random_walk= False)
    

    # rospy.loginfo("Result from NORMAL")
    # successfully_loaded = load_q_table(q_table_name="q_table_evader.txt", player_type="evader")
    # if successfully_loaded:
    #     test("evader", total_episodes= 100, episode_time_limit=90, allow_evader_manual_rescue=True, time_to_apply_action= 0.5, evader_random_walk= False)
    
    
    
    
    # rospy.spin()

 

if __name__ == "__main__":
    main()