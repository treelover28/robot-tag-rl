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

from activation_functions import sigmoid, relu, relu_derivative
from random_walking_agent import Random_Walking_Agent
from q_learning_agent import Simple_Q_Learning_Agent
from dqn_agent import DQN_Agent
from learning_plotter import Learning_Plotter
from mischallenous_functions import _get_permutations
from sklearn.preprocessing import normalize


# Game information
GAME_TIMEOUT = False
PURSUER_STATE_DISCRETIZED = None 
PURSUER_STATE_CONTINUOUS = None
PURSUER_POSITION = None 
PURSUER_STUCK = False


EVADER_STATE_DISCRETIZED = None 
EVADER_STATE_CONTINUOUS = None
EVADER_STUCK = False
EVADER_POSITION = None 
NUM_TIMES_EVADER_STUCK_IN_EPISODE = 0

EVADER_MIN_DISTANCE_TO_OBSTACLE = None
PURSUER_MIN_DISTANCE_TO_OBSTACLE = None 
DISTANCE_BETWEEN_PLAYERS = None

PURSUER_WAS_STUCK_BUT_RESCUED = False 
EVADER_WAS_STUCK_BUT_RESCUED = False 

RESCUE_PURSUER_FAILED = False
RESCUE_EVADER_FAILED = False

# for ros_plaza
# STARTING_LOCATIONS = [(0,1.2), (-2,1), (0,-1), (0,1.5), (0,-2), (-2,-1), (0.5,0), (-2,1.8),(1,0), (1,-2)]

# for ros 5 pillars map
STARTING_LOCATIONS = [(0,1), (-1,0), (0,-1), (1,0), (-1,-2), (-1,2)]

# for original ros map with all the pillars 
# STARTING_LOCATIONS = [(0.5,-0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (-1,-2), (-1,2)]

# for empty ros map with one pillar
# STARTING_LOCATIONS = [(0.5,-0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (-1,-2), (-1,2),(0,1), (-1,0), (0,-1), (1,0)]

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


# WIDTH OF TURTLEBOTS:
PURSUER_RADIUS = 1.0/8 # 4 Waffles one unit width wise
EVADER_RADIUS = 1.0/8 # 2 Burgers is roughly 1 waffle width wise



def get_opponent_position_rating(player_A, player_B):
    if player_A == PURSUER_POSITION:
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = PURSUER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE
    else:
        TRUE_SAFE_DISTANCE_FROM_OBSTACLE = EVADER_RADIUS + SAFE_DISTANCE_FROM_OBSTACLE
    
    
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
    DISTANCE_BETWEEN_PLAYERS =  np.linalg.norm(vector_player_to_opponent)
    
    if DISTANCE_BETWEEN_PLAYERS <= 0.3:
        rating = "Tagged"
    elif DISTANCE_BETWEEN_PLAYERS <= TRUE_SAFE_DISTANCE_FROM_OBSTACLE * 1.2:
        rating = "Close " + direction_rating
    else:
        rating = direction_rating

    return rating, DISTANCE_BETWEEN_PLAYERS, angle_degree

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
        if distance <= interval * 1.45:
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
    # LiDAR range readings
    ranges_data = message.ranges
    # maximum range of the LiDAR
    lidar_max_range = float(message.range_max)
    
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
        min_front = np.min([min_front, ranges_data[angle], lidar_max_range])
    for angle in upper_left_sector:
        min_upper_left = np.min([min_upper_left, ranges_data[angle], lidar_max_range])
    for angle in lower_left_sector:
        min_lower_left = np.min([min_lower_left, ranges_data[angle], lidar_max_range])
    for angle in upper_right_sector:
        min_upper_right = np.min([min_upper_right, ranges_data[angle], lidar_max_range])
    for angle in lower_right_sector:
        min_lower_right = np.min([min_lower_right, ranges_data[angle], lidar_max_range])
    

    if player_type == "pursuer":
        global PURSUER_STATE_DISCRETIZED 
        
        
        opp_rating, distance_between_player, angle_between_player = get_opponent_position_rating(PURSUER_POSITION, EVADER_POSITION)

        PURSUER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front, player_type), \
            "Upper Left" : get_distance_rating("Upper Left", min_upper_left, player_type), \
            "Upper Right": get_distance_rating("Upper Right", min_upper_right, player_type), \
            "Lower Left": get_distance_rating( "Lower Left", min_lower_left, player_type), \
            "Lower Right": get_distance_rating("Lower Right", min_lower_right, player_type), \
            "Opponent Position": opp_rating
        }
        global PURSUER_STATE_CONTINUOUS
        state = np.array([min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right, distance_between_player])/lidar_max_range 
        state = np.append(state, angle_between_player/360.0)
        # state = np.array([min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right, distance_between_player, angle_between_player/180.0 * pi])
        PURSUER_STATE_CONTINUOUS = np.reshape(state, (1, len(state)))

        global PURSUER_MIN_DISTANCE_TO_OBSTACLE
        all_direction = ["Front", "Upper Left", "Lower Left", "Upper Right", "Lower Right"]
        all_distances =  [ min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ]
        PURSUER_MIN_DISTANCE_TO_OBSTACLE= min(all_distances)

        global PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION
        index_ = all_distances.index(PURSUER_MIN_DISTANCE_TO_OBSTACLE)
        PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION = all_direction[index_]

        if verbose:
            rospy.loginfo("Pursuer's state: {}".format(PURSUER_STATE_DISCRETIZED))

    else:
        global EVADER_STATE_DISCRETIZED 

        opp_rating, distance_between_player, angle_between_player = get_opponent_position_rating(EVADER_POSITION, PURSUER_POSITION)

        EVADER_STATE_DISCRETIZED = {
            "Front": get_distance_rating("Front", min_front, player_type), \
            "Upper Left" : get_distance_rating("Upper Left", min_upper_left, player_type), \
            "Upper Right": get_distance_rating("Upper Right", min_upper_right, player_type), \
            "Lower Left": get_distance_rating( "Lower Left", min_lower_left, player_type), \
            "Lower Right": get_distance_rating("Lower Right", min_lower_right, player_type), \
            "Opponent Position": opp_rating
        }

        global EVADER_STATE_CONTINUOUS
        state = np.array([min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right, distance_between_player])/lidar_max_range 
        state = np.append(state, angle_between_player/360.0)
        # state = np.array([min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right, distance_between_player, angle_between_player/180.0 * pi])
        EVADER_STATE_CONTINUOUS = np.reshape(state, (1, len(state)))
       
    
        global EVADER_MIN_DISTANCE_TO_OBSTACLE 
        all_direction = ["Front", "Upper Left", "Lower Left", "Upper Right", "Lower Right"]
        all_distances =  [ min_front, min_upper_left, min_lower_left, min_upper_right, min_lower_right ]
        EVADER_MIN_DISTANCE_TO_OBSTACLE= min(all_distances)

        global EVADER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION
        index_ = all_distances.index(EVADER_MIN_DISTANCE_TO_OBSTACLE)
        EVADER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION = all_direction[index_]

        if verbose:
            rospy.loginfo("Evader's state: {}".format(EVADER_STATE_DISCRETIZED))


def get_robot_state(robot_type, state_type):
    if robot_type == "pursuer" and state_type == "discrete":
        return PURSUER_STATE_DISCRETIZED
    elif robot_type == "evader" and state_type == "discrete":
        return EVADER_STATE_DISCRETIZED
    elif robot_type == "pursuer" and state_type == "continuous":
        return PURSUER_STATE_CONTINUOUS
    else:
        return EVADER_STATE_CONTINUOUS


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

def _move_robot(player_type, translation_speed, angular_speed_degrees):
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


def robot_take_action(player_type, translational_speed, angular_speed, time_to_take_action):
    _move_robot(player_type, translational_speed, angular_speed)
    rospy.sleep(time_to_take_action)


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
        
        # spin to find opening
        if to_turn_left:
            _move_robot(robot_type,0, 60)
        else:
            _move_robot(robot_type, 0, -60)
       
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
    
    # stop robot once opening is in front
    _move_robot(robot_type, 0, 0)
    
    
def manual_reversal(robot_type, time_to_apply_action=1.5):
    if robot_type == "pursuer":
        robot_state = PURSUER_STATE_DISCRETIZED
    else:
        robot_state = EVADER_STATE_DISCRETIZED

    if robot_state["Opponent Position"] == "Tagged":
        return 

    close_ratings = ["Too Close", "Close"]
    
    translation_speed = -0.25
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
    
    _move_robot(robot_type, translation_speed, turn_angle)
    
    rospy.sleep(time_to_apply_action)
    # rospy.loginfo("Slept")


def spawn_robots():
    # spawn pursuers and evaders at different locations throughout the map 
    pursuer_position = None 
    evader_position = None 

    is_far_enough = False
    while (pursuer_position == evader_position or not is_far_enough):
        pursuer_position = STARTING_LOCATIONS[random.randint(0, len(STARTING_LOCATIONS) - 1)]
        evader_position = STARTING_LOCATIONS[random.randint(0, len(STARTING_LOCATIONS) - 1)]
        is_far_enough = np.linalg.norm(np.array(pursuer_position) - np.array(evader_position)) > 0.5
    
    set_robot_position("pursuer", pursuer_position)
    set_robot_position("evader", evader_position)


def get_game_information(information_name):
    # get in-game information
    if information_name == "pursuer_position":
        information =  PURSUER_POSITION
    elif information_name == "evader_position":
        information = EVADER_POSITION
    elif information_name == "pursuer_radius":
        information = PURSUER_RADIUS
    elif information_name == "evader_radius":
        information = EVADER_RADIUS
    elif information_name == "safe_distance_from_obstacle":
        information = SAFE_DISTANCE_FROM_OBSTACLE
    elif information_name == "pursuer_stuck":
        information = PURSUER_STUCK
    elif information_name == "evader_stuck":
        information = EVADER_STUCK
    elif information_name == "pursuer_was_stuck_but_rescued":
        information = PURSUER_WAS_STUCK_BUT_RESCUED
    elif information_name == "evader_was_stuck_but_rescued":
        information = EVADER_WAS_STUCK_BUT_RESCUED
    elif information_name == "pursuer_min_distance_to_obstacle":
        information = PURSUER_MIN_DISTANCE_TO_OBSTACLE
    elif information_name == "evader_min_distance_to_obstacle":
        information = EVADER_MIN_DISTANCE_TO_OBSTACLE
    elif information_name == "pursuer_min_distance_to_obstacle_direction":
        information = PURSUER_MIN_DISTANCE_TO_OBSTACLE_DIRECTION
    elif information_name == "game_timeout":
        information = GAME_TIMEOUT  
    elif information_name == "num_times_evader_stuck_in_episode":
        information = NUM_TIMES_EVADER_STUCK_IN_EPISODE
    elif information_name == "directional_states":
        information = DIRECTIONAL_STATES
    elif information_name == "distance_between_players":
        information = DISTANCE_BETWEEN_PLAYERS  
    else:
        information = None
    return information

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


def is_terminal_state(player_type, game_timeout, pursuer_stuck, evader_stuck, distance_between_players, verbose=True, allow_player_rescue = False):
        if distance_between_players <= 0.3: 
            is_terminal = True
            terminal_status = "Terminated because TAGGED. Pursuer Won"
        elif game_timeout:
            is_terminal = True
            terminal_status = "Terminated because game-timeot. Evader won"
        # if we are just training the pursuer, even if the evader gets stuck
        # we still let the pursuer run until it catches the evader, or gets stucks itself
        elif player_type == "pursuer" and pursuer_stuck and not allow_player_rescue:
            is_terminal = True
            terminal_status = "Terminated because pursuer is STUCK"
        # when training the evader, terminate when the evader gets stuck
        elif player_type == "evader" and evader_stuck and not allow_player_rescue:
            is_terminal = True
            terminal_status = "Terminated because evader is STUCK"
        else:
            is_terminal = False 
        
        if is_terminal and verbose:
            rospy.loginfo(terminal_status)
        
        return is_terminal

def train(train_type, pursuer_agent, evader_agent, total_episodes = 1000, starting_epsilon = 0.2, max_epsilon = 0.9, episode_time_limit = 30, time_to_apply_action=0.5, do_initial_test = False, allow_player_manual_rescue= False):
    
    if train_type not in ["pursuer", "evader"]:
        rospy.loginfo("Unrecognized train type. Either \"puruser\" or \"evader\"")
        return 
    
    current_episode = 0
    # accumulated_pursuer_reward = 0
    # accumulated_evader_reward = 0
    epsilon = starting_epsilon

    if train_type == "pursuer":
        player = "pursuer"
        opponent = "evader"
        player_agent = pursuer_agent
        opponent_agent = evader_agent
    else:
        player = "evader"
        opponent = "pursuer"
        player_agent = evader_agent
        opponent_agent = pursuer_agent
   
    # create a plotter to plot training curves and other metrics 
    plotter = Learning_Plotter(train_type= train_type, total_episodes = total_episodes, episode_time_limit=episode_time_limit, training_algorithm_name=player_agent.agent_algorithm, allow_player_manual_rescue=allow_player_manual_rescue)

    # keep track of best training and testing scores
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
    accumulated_num_stuck_by_evader = 0
    total_loss_dqn = 0.0
    total_Q = 0.0
    num_batches = 0.0
    while current_episode < total_episodes:
        if (PURSUER_STATE_DISCRETIZED is not None and EVADER_STATE_DISCRETIZED is not None):
            rospy.loginfo("Starting Episode {}".format(current_episode))
            print("*"*50)
 
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
                    test_reward, num_tagged, num_stuck, num_timeout = test("pursuer", pursuer_agent = pursuer_agent, evader_agent = evader_agent, total_episodes = 40, episode_time_limit=episode_time_limit, time_to_apply_action=time_to_apply_action, allow_evader_manual_rescue= True)
                elif train_type == "evader":
                    test_reward, num_tagged, num_stuck, num_timeout = test("evader", pursuer_agent = pursuer_agent, evader_agent = evader_agent, total_episodes = 40, episode_time_limit=episode_time_limit, time_to_apply_action=time_to_apply_action, allow_pursuer_manual_rescue= True)
                
                if test_reward > best_test_score:
                    # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                    # save q-table
                    player_agent.save_agent("{}_{}_best_testing.txt".format(player_agent.agent_algorithm, player))
                    best_test_score = test_reward
                
                # plot learning cureve with test rewards
                plotter.plot_learning_curve(plotter.test_curve, current_episode, test_reward)
                # plot tag curve, stuck curve and timeout curve
                plotter.plot_learning_curve(plotter.tag_curve, current_episode, num_tagged)
                plotter.plot_learning_curve(plotter.stuck_curve, current_episode, num_stuck)
                plotter.plot_learning_curve(plotter.timeout_curve, current_episode, num_timeout)
            
    
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
            
            global NUM_TIMES_EVADER_STUCK_IN_EPISODE
            NUM_TIMES_EVADER_STUCK_IN_EPISODE = 0

            is_terminal = False
            

            while(not is_terminal):
                # get distance between two agents
                distance_between_players = DISTANCE_BETWEEN_PLAYERS

                # get time elapsed so far 
                time_elapsed = (rospy.Time.now() - start_time) - time_spent_on_manual_rescue
                
                # check if the game has timeout
                if time_elapsed >= rospy.Duration(secs = episode_time_limit):
                    GAME_TIMEOUT = True
                
                global PURSUER_WAS_STUCK_BUT_RESCUED
                global EVADER_WAS_STUCK_BUT_RESCUED
            
                PURSUER_WAS_STUCK_BUT_RESCUED = False
                EVADER_WAS_STUCK_BUT_RESCUED = False

                # initialize a seperate thread to handle rescuing the opponent should it gets stuck
                # since we are training our player, we assume the opponent is already good
                # training episode would only terminate if the player gets stuck but not when the opponent gets stuck 
                pursuer_rescue_thread = threading.Thread(target=manual_rescue, args = ("pursuer", 1.0))
                evader_rescue_thread = threading.Thread(target=manual_rescue, args= ("evader", 1.0))
                
                # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 1.5 seconds
                if len(last_few_pursuer_positions) == int(1.5/time_to_apply_action):
                    PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state= get_robot_state("pursuer", "discrete"))
                    if PURSUER_STUCK and not GAME_TIMEOUT:
                        if train_type == "evader" or (train_type == "pursuer" and allow_player_manual_rescue): 
                            # if we are training the evader, the pursuer could manually reverse to rescue itself when stuck
                            # and resume chasing the evader
                            # rescue_pursuer_start_time = rospy.Time.now()
                            pursuer_rescue_thread.start()
                            last_few_pursuer_positions = []
                            # get new state after reversal
                            PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=get_robot_state("pursuer", "discrete"))
                    if len(last_few_pursuer_positions) != 0:
                        del last_few_pursuer_positions[0]
                    
                if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                    EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= get_robot_state("evader", "discrete"))
                    if EVADER_STUCK and not GAME_TIMEOUT:
                        NUM_TIMES_EVADER_STUCK_IN_EPISODE += 1
                        if (train_type == "pursuer" and  evader_agent.agent_algorithm != "Random-Walk") or (train_type == "evader" and allow_player_manual_rescue): 
                            # when training the pursuer, we have two modes
                            # we could either train it against a random-walking evader that has a reversal already coded in
                            # or we could train it against an evader which uses a q-table with no reversal actions.
                            # when we are training against the latter, we have to call the manual rescue thread should
                            # the evader gets stuck since it does not have the reversal action in its q-table
                            # rescue_evader_start_time = rospy.Time.now()
                            evader_rescue_thread.start()
                            last_few_evader_positions = []
                            # get new state after reversal
                            EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=get_robot_state("evader", "discrete"))
                    
                    if len(last_few_evader_positions) != 0:
                        del last_few_evader_positions[0]

                last_few_pursuer_positions.append(PURSUER_POSITION[:2])
                last_few_evader_positions.append(EVADER_POSITION[:2])
                

                while pursuer_rescue_thread.is_alive():
                    if train_type == "evader" and evader_agent.agent_algorithm != "Random-Walk":
                        # while waiting for opponent pursuer to rescue itself, the agent evader continues learninng
                        evader_agent.learn(epsilon = epsilon, time_to_apply_action = time_to_apply_action)
                    elif train_type == "pursuer":
                        evader_agent.follow_policy(time_to_apply_action=time_to_apply_action)
                    
                    PURSUER_WAS_STUCK_BUT_RESCUED = True
                    # rescue_stop_time = rospy.Time.now()
                    # time_spent_on_manual_rescue += (rescue_stop_time - rescue_pursuer_start_time)

                while evader_rescue_thread.is_alive():
                    if train_type == "pursuer":
                        # continue training pursuer while opponent evader rescue itself
                        pursuer_agent.learn(epsilon = epsilon, time_to_apply_action = time_to_apply_action)
                    elif train_type == "evader":
                        pursuer_agent.follow_policy(time_to_apply_action=time_to_apply_action)
                    
                    EVADER_WAS_STUCK_BUT_RESCUED = True

                    # evader_rescue_stop_time = rospy.Time.now()
                    # time_spent_on_manual_rescue += (evader_rescue_stop_time - rescue_evader_start_time)

                if not GAME_TIMEOUT and (RESCUE_PURSUER_FAILED or RESCUE_EVADER_FAILED):
                    # if failed to rescue, break out and restart episode
                    if RESCUE_EVADER_FAILED and RESCUE_PURSUER_FAILED:
                        rospy.loginfo("RESCUE FAILED FOR BOTH")
                    elif RESCUE_PURSUER_FAILED:
                        rospy.loginfo("RESCUE PURSUER FAILED")
                    else:
                        rospy.loginfo("RESCUE EVADER FAILED")
                    break


                opponent_decision_making_thread = threading.Thread(target = opponent_agent.follow_policy, args=(time_to_apply_action,))
                opponent_decision_making_thread.start()

                # get current state before agent learns and carry out action
                current_state_discretized = get_robot_state(train_type, "discrete")
                # have current robot train using their respective learning algorithm
                learning_report = player_agent.learn(epsilon = epsilon, time_to_apply_action = time_to_apply_action)
                
                if player_agent.agent_algorithm == "DQN":
                    reward, _ , _, turn_action, rmse_current_batch, avg_Q_current_batch = learning_report
                    # Accumulate total loss for Deep Q Network agent
                    if rmse_current_batch != None:
                        total_loss_dqn += rmse_current_batch
                        num_batches += 1
                        
                    if avg_Q_current_batch != None:
                        total_Q += avg_Q_current_batch
                else:
                    reward, _ , _, turn_action = learning_report
                    
                # accumulate rewards
                accumulated_reward += reward

                # accumulate number of times each scenarios happen to check for state-action convergence
                if "Left" in current_state_discretized["Opponent Position"]:
                    if turn_action >= 20:
                        num_times_go_left_opponent_is_left += 1
                    elif turn_action <= -20:
                        num_times_go_right_opponent_is_left += 1
                    else:
                        num_times_go_front_opponent_is_left += 1
                elif "Right" in current_state_discretized["Opponent Position"]:
                    if turn_action >= 20:
                        num_times_go_left_opponent_is_right += 1
                    elif turn_action <= -20:
                        num_times_go_right_opponent_is_right += 1
                    else:
                        num_times_go_front_opponent_is_right += 1
                elif "Front" in current_state_discretized["Opponent Position"]:
                    if turn_action >= 20:
                        num_times_go_left_opponent_is_front += 1
                    elif turn_action <= -20:
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
                
                # check if this is terminal state
                is_terminal = is_terminal_state(player_agent.agent_type, GAME_TIMEOUT, PURSUER_STUCK, EVADER_STUCK, distance_between_players, verbose=False, allow_player_rescue = allow_player_manual_rescue)
            
            # restart episode if manual rescue failed, don't count
            if not RESCUE_EVADER_FAILED and not RESCUE_PURSUER_FAILED:
                current_episode += 1

            # accumulate metrics
            if DISTANCE_BETWEEN_PLAYERS != None:
                accumulated_distance_between_players_at_end += DISTANCE_BETWEEN_PLAYERS
            
            # accumulate time the evader survived on the map so far
            accumulated_time_survived_by_evader += time_elapsed
            training_reward += accumulated_reward
            accumulated_num_stuck_by_evader += NUM_TIMES_EVADER_STUCK_IN_EPISODE

            if current_episode % 200 == 0:
                # plot learning curve using the average reward each 250 training episodes
                training_reward /= 200.0
                plotter.plot_learning_curve(plotter.learning_curve, current_episode, training_reward)

                if training_reward > best_train_score:
                    best_train_score = training_reward
                    # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                    # save q-table
                    player_agent.save_agent("{}_{}_best_training.txt".format(player_agent.agent_algorithm, player))
    
                # RESET TRAINING REWARD
                training_reward = 0.0
                
               
                if player_agent.agent_algorithm == "DQN":
                    # plot the mean squared loss per mini batch for every 250 episodes
                    # print( total_loss_dqn/(20.0))
                    plotter.plot_learning_curve(plotter.avg_loss_curve, current_episode, total_loss_dqn/num_batches)
                    total_loss_dqn = 0.0
                    
                    # plot the average Q-value norm for every 250 episodes
                    plotter.plot_learning_curve(plotter.avg_q_curve, current_episode, total_Q/num_batches)
                    total_Q = 0.0
                    
                    # reset number of mini-batches
                    num_batches = 0.0

                 # plot the average distance at terminal state for every 250 episodes
                if train_type == "pursuer":
                    plotter.plot_learning_curve(plotter.average_distance_at_terminal_curve, current_episode, accumulated_distance_between_players_at_end/200.0)
                    # reset
                    accumulated_distance_between_players_at_end = 0
                elif train_type == "evader" and not allow_player_manual_rescue:
                    plotter.plot_learning_curve(plotter.average_time_at_terminal_curve, current_episode, (accumulated_time_survived_by_evader/200.0).to_sec())
                    accumulated_time_survived_by_evader = rospy.Duration(secs = 0)
                else:
                    plotter.plot_learning_curve(plotter.num_evader_stuck_curve, current_episode, accumulated_num_stuck_by_evader/200.0)
                    accumulated_num_stuck_by_evader = 0
                

            if current_episode % 500 == 0:
                # plot state-action convergence graph
                # when proportion of action when opponent is to the left
                plotter.plot_learning_curve(plotter.go_left_when_opponent_left_curve, current_episode, float(num_times_go_left_opponent_is_left)/sum([num_times_go_front_opponent_is_left, num_times_go_left_opponent_is_left, num_times_go_right_opponent_is_left]))
                plotter.plot_learning_curve(plotter.go_right_when_opponent_left_curve, current_episode, float(num_times_go_right_opponent_is_left)/sum([num_times_go_front_opponent_is_left, num_times_go_left_opponent_is_left, num_times_go_right_opponent_is_left]))
                plotter.plot_learning_curve(plotter.go_front_when_opponent_left_curve, current_episode, float(num_times_go_front_opponent_is_left)/sum([num_times_go_front_opponent_is_left, num_times_go_left_opponent_is_left, num_times_go_right_opponent_is_left]))
                # when proportion of action when opponent is to the right
                plotter.plot_learning_curve(plotter.go_left_when_opponent_right_curve, current_episode, float(num_times_go_left_opponent_is_right)/sum([num_times_go_front_opponent_is_right, num_times_go_left_opponent_is_right, num_times_go_right_opponent_is_right]))
                plotter.plot_learning_curve(plotter.go_right_when_opponent_right_curve, current_episode, float(num_times_go_right_opponent_is_right)/sum([num_times_go_front_opponent_is_right, num_times_go_left_opponent_is_right, num_times_go_right_opponent_is_right]))
                plotter.plot_learning_curve(plotter.go_front_when_opponent_right_curve, current_episode, float(num_times_go_front_opponent_is_right)/sum([num_times_go_front_opponent_is_right, num_times_go_left_opponent_is_right, num_times_go_right_opponent_is_right]))
                # when proportion of action when opponent is in front
                plotter.plot_learning_curve(plotter.go_left_when_opponent_front_curve, current_episode, float(num_times_go_left_opponent_is_front)/sum([num_times_go_front_opponent_is_front, num_times_go_left_opponent_is_front, num_times_go_right_opponent_is_front]))
                plotter.plot_learning_curve(plotter.go_right_when_opponent_front_curve, current_episode, float(num_times_go_right_opponent_is_front)/sum([num_times_go_front_opponent_is_front, num_times_go_left_opponent_is_front, num_times_go_right_opponent_is_front]))
                plotter.plot_learning_curve(plotter.go_front_when_opponent_front_curve, current_episode, float(num_times_go_front_opponent_is_front)/sum([num_times_go_front_opponent_is_front, num_times_go_left_opponent_is_front, num_times_go_right_opponent_is_front]))
                
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
                plotter.savefig("{}_{}_{}_episodes".format(player_agent.agent_algorithm, train_type, current_episode), dpi=100)
            
            # save q-table every training episode
            player_agent.save_agent("{}_{}.txt".format(player_agent.agent_algorithm, player))
            # notify user that Q-table has been saved
            rospy.loginfo("Saved Agent_{}".format(player))


def test(player, pursuer_agent, evader_agent, total_episodes = 2, episode_time_limit=30, time_to_apply_action = 0.5, allow_pursuer_manual_rescue= False, allow_evader_manual_rescue = False):
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
                opponent = "evader"
                player_agent = pursuer_agent
                opponent_agent = evader_agent
            else:
                opponent = "pursuer"
                player_agent = evader_agent
                opponent_agent =  pursuer_agent
            
            current_state = get_robot_state(player, "discrete")
        
        # keeps track of how much time is left in current round
        start_time = rospy.Time.now()
        # time_elapsed = rospy.Duration(secs=0)
        
        global GAME_TIMEOUT
        GAME_TIMEOUT = False
        
        # keep track of total time spent on rescuing the robot
        # rescue time will not count toward game time
        time_spent_on_manual_rescue = rospy.Duration(secs=0)
        is_terminal = False
        
        while(not is_terminal):
            
            global PURSUER_WAS_STUCK_BUT_RESCUED
            global EVADER_WAS_STUCK_BUT_RESCUED
            
            PURSUER_WAS_STUCK_BUT_RESCUED = False
            EVADER_WAS_STUCK_BUT_RESCUED = False
            
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
                PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=get_robot_state("pursuer", "discrete"))
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
                        PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state=get_robot_state("pursuer", "discrete"))
                if len(last_few_pursuer_positions) != 0:
                    del last_few_pursuer_positions[0]
                
            if len(last_few_evader_positions) == int(1.5/time_to_apply_action):
                EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= get_robot_state("evader", "discrete"))
                if EVADER_STUCK and not GAME_TIMEOUT:
                    num_evader_stuck += 1
                    if (player == "pursuer" and evader_agent.agent_algorithm != "Random-Walk") or (player == "evader" and allow_evader_manual_rescue): 
                        # when training the pursuer, we have two modes
                        # we could either train it against a random-walking evader that has a reversal already coded in
                        # or we could train it against an evader which uses a q-table with no reversal actions.
                        # when we are training against the latter, we have to call the manual rescue thread should
                        # the evader gets stuck since it does not have the reversal action in its q-table
                        evader_rescue_start_time = rospy.Time.now()
                        evader_rescue_thread.start()
                        last_few_evader_positions = []
                        # get new state after reversal
                        EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state=get_robot_state("evader", "discrete"))
                
                if len(last_few_evader_positions) != 0:
                    # rospy.loginfo(EVADER_STUCK)
                    del last_few_evader_positions[0]

            last_few_pursuer_positions.append(PURSUER_POSITION[:2])
            last_few_evader_positions.append(EVADER_POSITION[:2])

            # wait for the rescue threads to join
            while pursuer_rescue_thread.is_alive():
                # while waiting for pursuer to unstuck itself, continue moving the evader
                
                evader_agent.follow_policy(time_to_apply_action=time_to_apply_action,  verbose = (player == "evader"))
                # move_robot("evader", 0,0)
                # pursuer_rescue_thread.join()
                # pursuer_rescue_stop_time = rospy.Time.now()
                # time_spent_on_manual_rescue += (pursuer_rescue_stop_time - pursuer_rescue_start_time)
                PURSUER_WAS_STUCK_BUT_RESCUED = True
            
            while evader_rescue_thread.is_alive():
                # while waiting for evader to unstuck itself, continue moving the pursuer
                pursuer_agent.follow_policy(time_to_apply_action=time_to_apply_action, verbose = (player == "pursuer"))
                # evader_rescue_thread.join()
                # evader_rescue_stop_time = rospy.Time.now()
                # time_spent_on_manual_rescue += (evader_rescue_stop_time - evader_rescue_start_time)
                EVADER_WAS_STUCK_BUT_RESCUED = True
            
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
            opponent_decision_making_thread = threading.Thread(target = opponent_agent.follow_policy, args=(time_to_apply_action,))
            opponent_decision_making_thread.start()
            
            # player's decision making
            player_agent.follow_policy(time_to_apply_action=time_to_apply_action,  verbose = False)

            # wait for opponent thread to finish before moving on
            opponent_decision_making_thread.join()
            
            # observe rewards of new state
            current_state = get_robot_state(player, "discrete")
            current_state_reward, _ = player_agent.reward_function(current_state, verbose=False)
            accumulated_reward += current_state_reward
            
            current_state = get_robot_state(player, "discrete")
            if not GAME_TIMEOUT and current_state["Opponent Position"] == "Tagged":
                num_tagged += 1

            is_terminal = is_terminal_state(player_type=player, game_timeout= GAME_TIMEOUT ,pursuer_stuck= PURSUER_STUCK, evader_stuck= EVADER_STUCK, \
                                    distance_between_players = DISTANCE_BETWEEN_PLAYERS, verbose=True)

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
    

    
    # pursuer_agent = Simple_Q_Learning_Agent("pursuer", 0.2 , 0.8 ,[],[], get_robot_state_discretized, robot_take_action, get_game_information)
    action_space = []
    translation_actions= [0.1, 0.2]
    rotational_actions = [-60, -40, -20, -5, 0, 5, 20, 40, 60]
    _get_permutations(0, [translation_actions, rotational_actions] ,list(), 2, action_space)

    pursuer_agent = DQN_Agent(agent_type = "pursuer", input_layer_size = 7, output_layer_size = len(translation_actions)* len(rotational_actions), hidden_layers = [16,16,16,16],\
        action_space = action_space, activation_function = relu, activation_function_derivative = relu_derivative, num_steps_to_update_network = 2000, batch_size = 64,
        learning_rate = 0.00005, discount_factor = 0.95, get_agent_state_function = get_robot_state, agent_take_action_function = robot_take_action, get_game_information= get_game_information)

    
    # pursuer_agent = DQN_Agent.load_agent("DQN_pursuer_best_testing.txt")
    # pursuer_agent = Simple_Q_Learning_Agent("pursuer", 0.2, 0.8, [], [], get_robot_state_discretized, robot_take_action, get_game_information)
    # successfully_loaded_pursuer = pursuer_agent.load_agent("q_table_pursuer_best_training_on_ros_map_against_good_evader_90%.txt")

    evader_agent = Random_Walking_Agent("evader", get_robot_state, robot_take_action, get_game_information, 0.20)
    # successfully_loaded_evader = True
    # if successfully_loaded_pursuer and successfully_loaded_evader:
    #     test("pursuer", pursuer_agent, evader_agent, total_episodes=100, episode_time_limit= 90, time_to_apply_action=0.25, allow_evader_manual_rescue= True, allow_pursuer_manual_rescue=True)
   
   
    # evader_agent = Simple_Q_Learning_Agent("evader", 0.2, 0.8,[],[], get_robot_state, robot_take_action, get_game_information)
    # successfully_loaded_evader = evader_agent.load_agent("q_table_evader_best_testing.txt")

    train("pursuer", pursuer_agent, evader_agent, total_episodes= 20000, starting_epsilon=0.4, max_epsilon=0.9, episode_time_limit=45, time_to_apply_action= 0.5, do_initial_test=False, allow_player_manual_rescue=False)
    # if successfully_loaded_pursuer and successfully_loaded_evader:
    # test("pursuer", pursuer_agent, evader_agent, total_episodes=100, episode_time_limit= 90, time_to_apply_action=0.25, allow_evader_manual_rescue= True, allow_pursuer_manual_rescue=True)
    
    # move_robot("evader", 0.0,0)
    
    # rospy.spin()



if __name__ == "__main__":
    main()