#! /usr/bin/python



import rospy
import os.path
import cPickle as pickle

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

PURSUER_POSITION = None 
PURSUER_STUCK = False


EVADER_POSITION = None 
EVADER_STUCK = False
NUM_TIMES_EVADER_STUCK_IN_EPISODE = 0
NUM_TIMES_PURSUER_STUCK_IN_EPISODE = 0

DISTANCE_BETWEEN_PLAYERS = None

PURSUER_WAS_STUCK_BUT_RESCUED = False 
EVADER_WAS_STUCK_BUT_RESCUED = False 

RESCUE_PURSUER_FAILED = False
RESCUE_EVADER_FAILED = False

# for ros_plaza
# STARTING_LOCATIONS = [(0,1.2), (-2,1), (0,-1), (0,1.5), (0,-2), (-2,-1), (0.5,0), (-2,1.8),(1,0), (1,-2)]

# for ros 5 pillars map
# STARTING_LOCATIONS = [(0,1), (-1,0), (0,-1), (1,0), (-1,-2), (-1,2)]

# for original ros map with all the pillars 
STARTING_LOCATIONS = [(0.5,-0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (-1,-2), (-1,2)]

# for empty ros map with one pillar
# STARTING_LOCATIONS = [(0.5,-0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (-1,-2), (-1,2),(0,1), (-1,0), (0,-1), (1,0)]

# State Space Hyperparameters
SAFE_DISTANCE_FROM_OBSTACLE = 0.3

DIRECTIONAL_STATES = ["Front", "Upper Left", "Upper Right", "Lower Left", "Lower Right","Opponent Position"]
FRONT_RATINGS = ["Close", "OK", "Far"]
UPPER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
UPPER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
LOWER_LEFT_RATINGS = ["Too Close", "Close", "OK", "Far"]
LOWER_RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far"]
OPPONENT_RATINGS = ["Close Left", "Left", "Close Front", "Front", "Right", "Close Right", "Bottom", "Close Bottom", "Tagged"]


EVADER_LIDAR_READINGS = None
EVADER_LIDAR_MAX_RANGE = None
PURSUER_LIDAR_READINGS = None 
PURSUER_LIDAR_MAX_RANGE = None

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


def set_lidar_readings(message,args):
    '''
    Get a message from ROS Subscriber to the Scan topic to get an agent's LIDAR readings.
    '''
    # LiDAR range readings
    ranges_data = message.ranges
    # maximum range of the LiDAR
    lidar_max_range = float(message.range_max)

    player_type = args["player_type"]
    verbose = args["verbose"]
    
    if player_type == "pursuer":
        global PURSUER_LIDAR_READINGS
        global PURSUER_LIDAR_MAX_RANGE
        PURSUER_LIDAR_READINGS = [min(reading, lidar_max_range) for reading in ranges_data]
        PURSUER_LIDAR_MAX_RANGE = lidar_max_range
    else:
        global EVADER_LIDAR_READINGS
        global EVADER_LIDAR_MAX_RANGE
        EVADER_LIDAR_READINGS = [min(reading, lidar_max_range) for reading in ranges_data]
        EVADER_LIDAR_MAX_RANGE = lidar_max_range

def get_lidar_readings(player_type):
    if player_type == "pursuer":
        return PURSUER_LIDAR_READINGS, PURSUER_LIDAR_MAX_RANGE
    else:
        return EVADER_LIDAR_READINGS, EVADER_LIDAR_MAX_RANGE


def get_robot_location(message, args):
    '''
    Get a message from ROS Subscriber to the Odometry topic to get an agent's position/location on the map.
    '''
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
    
    if PURSUER_POSITION != None and EVADER_POSITION != None:
        global DISTANCE_BETWEEN_PLAYERS
        DISTANCE_BETWEEN_PLAYERS = np.linalg.norm(np.array(PURSUER_POSITION[:3]) - np.array(EVADER_POSITION[:3]))


def _move_robot(player_type, translation_speed, angular_speed_degrees):
    """ Receive a linear speed and an angular speed (degrees/second), craft a Twist message,
    and send it to the /cmd_vel  topic on ROS to make the robot move.
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
    '''
    Function to communicate with Gazebo to spawn the agent at a specific pose.
    The pose parameter is simple (x,y) tuple specifying the agent's spawn coordinate.
    '''
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


def manual_rescue(agent, time_to_apply_action = 0.5, verbose = False):
    '''
    This function attempts to rescue/unstuck an agent should it collides with a nearby obstacles by first reversing in an approriate direction
    and spinning to reorient the agent toward an open space.
    '''
    robot_state = agent.get_current_state_discrete()
    
    # don't rescue if it gets tagged before
    if robot_state["Opponent Position"] == "Tagged":
        return

    manual_reversal(agent,time_to_apply_action=time_to_apply_action)
    manual_reorientation(agent, verbose= verbose)
    

def manual_reorientation(agent, time_to_apply_action=0.5, rescue_timeout_after_n_seconds = 10, verbose = False):
    '''
    This function reorients/rotates the agent to find an opening in front of it. This function is a part of the
    manual_rescue function which attempts to unstuck a stuck agent.
    '''
    robot_state = agent.get_current_state_discrete()
    

    to_turn_left = (robot_state["Opponent Position"] in ["Left", "Close Left"] and agent.agent_type == "pursuer") or\
                   (robot_state["Opponent Position"] in ["Right", "Close Right"] and agent.agent_type == "evader")

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
            _move_robot(agent.agent_type,0, 60)
        else:
            _move_robot(agent.agent_type, 0, -60)
       
        rospy.sleep(time_to_apply_action)
        
        # fetch new robot state
        robot_state = agent.get_current_state_discrete()
        
        if verbose:
            rospy.loginfo(rescue_status)

        # if the robot gets tagged while rescuing itself, stop the rescue
        if robot_state["Opponent Position"] == "Tagged":
            return 

    if rescue_timeout:
        if agent.agent_type == "pursuer":
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
    _move_robot(agent.agent_type, 0, 0)
    
    
def manual_reversal(agent, time_to_apply_action=1.5):
    '''
    This function deterministically take control of the agent and tries to reverse it.
    This function is a part of the manual_rescue function which attempts to unstuck a stuck agent.
    '''
    robot_state = agent.get_current_state_discrete()

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
    elif agent.agent_type == "pursuer" and robot_state["Opponent Position"] in ["Left", "Close Left"]:
        # rospy.loginfo("Right turn while reversing")    
        # right turn while reversing so pursuer could face the evader
        turn_angle = 60
    elif agent.agent_type == "pursuer" and robot_state["Opponent Position"] in ["Right", "Close Right"]:
        # left turn while reversing so pursuer could face the evader to the right
        # rospy.loginfo("Right turn while reversing")   
        turn_angle = -60
    elif agent.agent_type == "evader" and robot_state["Opponent Position"] in ["Left", "Close Left"]:
        # left turn while reversing so evader could face AWAY from the pursuer to the left
        turn_angle = -60
    else:
        turn_angle = 60
    
    _move_robot(agent.agent_type, translation_speed, turn_angle)
    
    rospy.sleep(time_to_apply_action)


def spawn_robots():
    '''
    This function spawns the pursuer and evader at different locations throughout the map randomly sampled from a
    predefined list of possible starting locations.
    '''
    
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
    '''
    This function is used to pass game information to the agents
    '''
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
    '''
    This function checks if an agent is stuck given a list of the agent's past few positions/poses
    '''
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
        is_in_same_place = changes_in_x < 0.025 and changes_in_y < 0.025

        # only check if robot's front is stuck, since if its side is stuck, it could rescue itself by turning the opposite direction
        # is_near_obstacle = robot_state["Front"] == "Close" 
        is_near_obstacle = True
        # the robot is consider stuck of it is near an obstacle and hasn't changed position in a while
        is_stuck = is_near_obstacle and is_in_same_place
    return is_stuck


def is_terminal_state(player_type, game_timeout, pursuer_stuck, evader_stuck, distance_between_players, verbose=True, allow_player_rescue = False):
    '''
    This function takes if agent's current state is a terminal state:
        - If the distance between two players are within a small threshold (0.3 units), we count it as a TAGGED terminal state
        - If we have reached the episode time limit.
        - If the agent gets stuck AND is not allow to rescue itself

    '''
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

def train(train_type, pursuer_agent, evader_agent, total_episodes = 10000, starting_epsilon = 0.2, max_epsilon = 0.9, epsilon_annealing_rate = 1.0001, episode_time_limit = 30, time_to_apply_action=0.5, do_initial_test = False, allow_player_manual_rescue= False):
    '''
    The train() method takes in an pursuer agent object, an evader agent object and train the agent specified by train_type parameter against the other. 
    
    Parameters:

    Training-Mode parameters
        - train_type: takes value of either "pursuer" or "evader". Method will return if invalid train_type value is given
        - pursuer_agent: takes an pursuing Agent object.
        - evader_agent: takes an pursuing Agent object.
        - total_episodes (optional): number of episodes to train the agents. Defauls to 10000 if not specified.
        - episode_time_limit (optional): maximum time limit (in seconds) of a training episode/game. Defauts to 30 seconds.
        - time_to_apply_action (optional): number of seconds to let the agent carry out its selected action. Deaults to 0.5 second(s)
        - do_initial_test (optional): Whether or not to do an intial test at the beginning of training. Not very helpful if we are training the agent from scratch. Defaults to False
        - allow_player_manual_rescue (optional) = Whether or not to allow the agent we are training to rescue itself. Defaults to False


    We use annealing epsilon to progressively makes the agent being trained exploits more of its knowledge as it learns about the game
        - starting_epsilon (optional): defaults to 0.2 
        - max_epsilon (optional): defaults to 0.9
        - epsilon_annealing_rate: defaults to 1.0001
    
    '''
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
        
        pursuer_state_discrete = pursuer_agent.get_current_state_discrete()
        evader_state_discrete = evader_agent.get_current_state_discrete()
        
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
            # code to train appropriate player
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
        

        # uses epsilon annealing
        # steadily increase epsilon until we reach a fixed max epsilon
        epsilon = min(epsilon * epsilon_annealing_rate, max_epsilon)

        accumulated_reward = 0
        

        # episode also terminates if we have reached the time limit

        global GAME_TIMEOUT
        GAME_TIMEOUT = False

        start_time = rospy.Time.now()
        time_elapsed = rospy.Duration(secs=0)
        time_spent_on_manual_rescue = rospy.Duration(secs=0)
        
        # the user is given an option whether or not an agent hitting an obstacle
        # is considered a terminal state (in the case we allow them to do manual rescue/reversal)
        # these two global variables keep track of how many time each agent collides with the obstacle
        global NUM_TIMES_EVADER_STUCK_IN_EPISODE
        NUM_TIMES_EVADER_STUCK_IN_EPISODE = 0

        global NUM_TIMES_PURSUER_STUCK_IN_EPISODE
        NUM_TIMES_PURSUER_STUCK_IN_EPISODE = 0

        is_terminal = False
        
        # while training episode hasn't ended yet
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

            # initialize a seperate thread to allow opponent to rescue itself should it gets stuck
            # since we are training our player, we assume the opponent is already good
            # thus, training episode would only terminate if the player gets stuck but not when the opponent gets stuck 
            pursuer_rescue_thread = threading.Thread(target=manual_rescue, args = (pursuer_agent, 1.5))
            evader_rescue_thread = threading.Thread(target=manual_rescue, args= (pursuer_agent, 1.5))
            
            # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 1.5 seconds
            # Code logic: the following code checks if the pursuer has been stuck in the same location for the past 1.5 seconds
            if len(last_few_pursuer_positions) >= int(1.5/time_to_apply_action):
                PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state= pursuer_agent.get_current_state_discrete())
                if PURSUER_STUCK and not GAME_TIMEOUT:
                    # this global variable is only meaningful if we allow the pursuer to rescue itself during training
                    NUM_TIMES_PURSUER_STUCK_IN_EPISODE += 1
                    # if we are training the evader, the pursuing opponent could manually reverse to rescue itself when stuck
                    # OR
                    # if we are training the pursuer and we allow the pursuer to rescue ifself
                    if train_type == "evader" or (train_type == "pursuer" and allow_player_manual_rescue): 
                        # rescue_pursuer_start_time = rospy.Time.now()
                        pursuer_rescue_thread.start()
                        last_few_pursuer_positions = []
                        # get new state after reversal
                        PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state= pursuer_agent.get_current_state_discrete())
                # remove oldest position of evader
                if len(last_few_pursuer_positions) != 0:
                    del last_few_pursuer_positions[0]
            
            # Code logic: the following code checks if the evader has been stuck in the same location for the past 1.5 seconds    
            if len(last_few_evader_positions) >= int(1.5/time_to_apply_action):
                EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= evader_agent.get_current_state_discrete())
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
                        EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= evader_agent.get_current_state_discrete())
                # remove oldest position of evader
                if len(last_few_evader_positions) != 0:
                    del last_few_evader_positions[0]

     
            # add latest (x,y) coordinates of the two agents to keep track of their locations
            # and help check whether they are stuck
            last_few_pursuer_positions.append(PURSUER_POSITION[:2])
            last_few_evader_positions.append(EVADER_POSITION[:2])
            

            while pursuer_rescue_thread.is_alive():
                if train_type == "evader" and evader_agent.agent_algorithm != "Random-Walk":
                    # while waiting for opponent pursuer to rescue itself, the agent evader continues learninng
                    evader_agent.learn(epsilon = epsilon, time_to_apply_action = time_to_apply_action)
                elif train_type == "pursuer":
                    # else if we are training the pursuer and allow it to rescue itself, the opponent evader keeps following its policy
                    evader_agent.follow_policy(time_to_apply_action=time_to_apply_action)
                PURSUER_WAS_STUCK_BUT_RESCUED = True
               

            while evader_rescue_thread.is_alive():
                if train_type == "pursuer":
                    # continue training pursuer while opponent evader rescue itself
                    pursuer_agent.learn(epsilon = epsilon, time_to_apply_action = time_to_apply_action)
                elif train_type == "evader":
                     # else if we are training the evader and allow it to rescue itself, the opponent pursuer keeps following its policy
                    pursuer_agent.follow_policy(time_to_apply_action=time_to_apply_action)
                EVADER_WAS_STUCK_BUT_RESCUED = True


            if not GAME_TIMEOUT and (RESCUE_PURSUER_FAILED or RESCUE_EVADER_FAILED):
                # if failed to rescue, break out and restart episode
                if RESCUE_EVADER_FAILED and RESCUE_PURSUER_FAILED:
                    rospy.loginfo("RESCUE FAILED FOR BOTH")
                elif RESCUE_PURSUER_FAILED:
                    rospy.loginfo("RESCUE PURSUER FAILED")
                else:
                    rospy.loginfo("RESCUE EVADER FAILED")
                break

            # starts opponent decision making thread
            # so the main thread controls the behavior of the agent being trained
            # and the other thread makes the opponent to follow its policy
            # THIS IS NOT A THREAD BOMB since agent.follow_policy() does not call any further thread
            # and terminate after select an action (for 1 time step)
            opponent_decision_making_thread = threading.Thread(target = opponent_agent.follow_policy, args=(time_to_apply_action,))
            opponent_decision_making_thread.start()

            # get current state before agent learns and carry out action
            current_state_discretized = player_agent.get_current_state_discrete()

            # have current robot train using their respective learning algorithm (Q-Learning, DQN)
            # the agent.learn() method returns a tuple of different metrics to evaluate the agent's learning.
            # please refer to the DQN paper by V.Mihn to understand what the metrics means
            learning_report = player_agent.learn(epsilon = epsilon, time_to_apply_action = time_to_apply_action)
            
            # unpack the metrics 
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

            # accumulate number of times each scenario happen
            # this is to check for state-action convergence and for plotting purposes
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
            
        
            # wait for opponent decision-thread to finish before continuing executing code
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

        # plot learning curve using the average reward each 200 training episodes
        if current_episode % 200 == 0:
            training_reward /= 200.0
            plotter.plot_learning_curve(plotter.learning_curve, current_episode, training_reward)

            if training_reward > best_train_score:
                best_train_score = training_reward
                # save the policy into a seperate Q-table everytime it achieve a high on the testing phase
                # save q-table
                player_agent.save_agent("{}_{}_best_training.txt".format(player_agent.agent_algorithm, player))

            # RESET TRAINING REWARD
            training_reward = 0.0
            
            # if we are training the agent via Deep Q-Networks
            if player_agent.agent_algorithm == "DQN":
                # plot the MSE per mini batch averaged over 200 episodes
                plotter.plot_learning_curve(plotter.avg_loss_curve, current_episode, total_loss_dqn/num_batches)
                total_loss_dqn = 0.0
                
                # plot the average Q-value norm averaged over 200 episodes
                plotter.plot_learning_curve(plotter.avg_q_curve, current_episode, total_Q/num_batches)
                total_Q = 0.0
                
                # reset number of mini-batches
                num_batches = 0.0

            # plot the average distance at terminal state for every 250 episodes
            if train_type == "pursuer":
                plotter.plot_learning_curve(plotter.average_distance_at_terminal_curve, current_episode, accumulated_distance_between_players_at_end/200.0)
                # reset accumulating variables
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


def test(player_to_test, pursuer_agent, evader_agent, total_episodes = 2, episode_time_limit=30, time_to_apply_action = 0.5, allow_pursuer_manual_rescue= False, allow_evader_manual_rescue = False):
    '''
    The test() method takes in an pursuer agent object, an evader agent object and test the agent specified by player_to_test parameter against the other.
    During testing, the epsilon is raised 0.95 to encourage high exploitation but still leaves room for a little exploration.
    
    Parameters:
        - player_to_test: takes value of either "pursuer" or "evader". Method will return if invalid value is given.
        - pursuer_agent: takes an pursuing Agent object.
        - evader_agent: takes an pursuing Agent object.
        - total_episodes (optional): number of episodes to test the agents. Defauls to 2 if not specified.
        - episode_time_limit (optional): maximum time limit (in seconds) of a testing episode/game. Defauts to 30 seconds.
        - time_to_apply_action (optional): number of seconds to let the agent carry out its selected action. Deaults to 0.5 second(s)
        - allow_pursuer_manual_rescue (optional) = Whether or not to allow the pursuer to rescue itself. Defaults to False.
        - allow_evader_manual_rescue (optional) = Whether or not to allow the pursuer to rescue itself. Defaults to False.
    '''
    
    
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
            if player_to_test == "pursuer":
                opponent = "evader"
                player_agent = pursuer_agent
                opponent_agent = evader_agent
            else:
                opponent = "pursuer"
                player_agent = evader_agent
                opponent_agent =  pursuer_agent
            
            current_state = player_agent.get_current_state_discrete()
        
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
            evader_rescue_thread = threading.Thread(target=manual_rescue, args = (evader_agent, 1.5))
            pursuer_rescue_thread = threading.Thread(target=manual_rescue, args = (pursuer_agent, 1.5))

            # don't count time to manually rescue players as part of game time
            time_elapsed = (rospy.Time.now() - start_time) - time_spent_on_manual_rescue
            
            # check if the game has timeout
            if time_elapsed >= rospy.Duration(secs = episode_time_limit):
                GAME_TIMEOUT = True
                num_timeout += 1

            # check if robots are stuck, the robot is considered stuck if it has been in the same location for >= 1.50 seconds
            if len(last_few_pursuer_positions) >= int(1.5/time_to_apply_action):
                PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state= pursuer_agent.get_current_state_discrete())
                if PURSUER_STUCK and not GAME_TIMEOUT:
                    num_pursuer_stuck += 1
                    # starts rescue thread if we are either
                    # testing how the robot performs with manual rescue on
                    # or if we are testing the evader against an good, adversarial pursuer
                    if (player_to_test == "pursuer" and allow_pursuer_manual_rescue) or (player_to_test == "evader"):
                        # pursuer_rescue_start_time = rospy.Time.now()
                        pursuer_rescue_thread.start()
                        last_few_pursuer_positions = []
                        # get new state after reversal
                        PURSUER_STUCK = is_stuck(last_few_pursuer_positions, robot_state= pursuer_agent.get_current_state_discrete())
                if len(last_few_pursuer_positions) != 0:
                    del last_few_pursuer_positions[0]
                
            if len(last_few_evader_positions) >= int(1.5/time_to_apply_action):
                EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= evader_agent.get_current_state_discrete())
                if EVADER_STUCK and not GAME_TIMEOUT:
                    num_evader_stuck += 1
                    if (player_to_test == "pursuer" and evader_agent.agent_algorithm != "Random-Walk") or (player_to_test == "evader" and allow_evader_manual_rescue): 
                        # when testing the pursuer, we have two modes
                        # we could either test it against a random-walking evader that has a reversal already coded in
                        # or we could test it against an evader which uses a q-table with no reversal actions.
                        # when we are testing against the latter, we have to call the manual rescue thread should
                        # the evader gets stuck since it does not have the reversal action in its q-table
                        # evader_rescue_start_time = rospy.Time.now()
                        evader_rescue_thread.start()
                        last_few_evader_positions = []
                        # get new state after reversal
                        EVADER_STUCK = is_stuck(last_few_evader_positions, robot_state= evader_agent.get_current_state_discrete())
                
                if len(last_few_evader_positions) != 0:
                    # rospy.loginfo(EVADER_STUCK)
                    del last_few_evader_positions[0]

            last_few_pursuer_positions.append(PURSUER_POSITION[:2])
            last_few_evader_positions.append(EVADER_POSITION[:2])

            # wait for the rescue threads to join
            while pursuer_rescue_thread.is_alive():
                # while waiting for pursuer to unstuck itself, continue moving the evader
                evader_agent.follow_policy(time_to_apply_action=time_to_apply_action,  verbose = (player_to_test == "evader"))
                PURSUER_WAS_STUCK_BUT_RESCUED = True
            
            while evader_rescue_thread.is_alive():
                # while waiting for evader to unstuck itself, continue moving the pursuer
                pursuer_agent.follow_policy(time_to_apply_action=time_to_apply_action, verbose = (player_to_test == "pursuer"))
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
            current_state = player_agent.get_current_state_discrete()
            current_state_reward, _ = player_agent.reward_function(current_state, verbose=False)
            accumulated_reward += current_state_reward
            
            if not GAME_TIMEOUT and current_state["Opponent Position"] == "Tagged":
                num_tagged += 1

            is_terminal = is_terminal_state(player_type=player_to_test, game_timeout= GAME_TIMEOUT ,pursuer_stuck= PURSUER_STUCK, evader_stuck= EVADER_STUCK, \
                                    distance_between_players = DISTANCE_BETWEEN_PLAYERS, verbose=True)

        if not RESCUE_EVADER_FAILED and not RESCUE_PURSUER_FAILED:    
            current_episode += 1

    # print diagnostic test phase details   
    rospy.loginfo("\nTEST PHASE DETAIL\n")    
    rospy.loginfo("TEST PHASE: {} times pursuer tagged the evader out of {} testing rounds".format(num_tagged, total_episodes))
    if player_to_test == "pursuer" and allow_pursuer_manual_rescue:
        rospy.loginfo("TEST PHASE: {} times the pursuer have to use the manual rescuing function to unstuck itself out of {} testing rounds".format(num_pursuer_stuck, total_episodes))
    elif player_to_test == "pursuer":
        rospy.loginfo("TEST PHASE: {} times the pursuer got stuck out of {} testing rounds".format(num_pursuer_stuck, total_episodes))
    elif player_to_test == "evader" and allow_pursuer_manual_rescue:
        rospy.loginfo("TEST PHASE: {} times the evader have to use the manual rescuing function to unstuck itself out of {} testing rounds".format(num_evader_stuck, total_episodes))
    else:
        rospy.loginfo("TEST PHASE: {} times the evader got stuck out of {} testing rounds".format(num_evader_stuck, total_episodes))
    rospy.loginfo("TEST PHASE: {} times the evader survived the match against the pursuer (timeout) out of {} testing rounds".format(num_timeout, total_episodes))
    rospy.loginfo("-"*50)
    
    if player_to_test == "pursuer":
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
    PURSUER_SCAN_SUBSCRIBER = rospy.Subscriber("/pursuer/scan", LaserScan, callback=set_lidar_readings, callback_args={"player_type" : "pursuer" ,"verbose": False})
    # repeat with evader
    EVADER_SCAN_SUBSCRIBER = rospy.Subscriber("/evader/scan", LaserScan, callback=set_lidar_readings, callback_args={"player_type" : "evader" ,"verbose": False})

    # get players 's positions
    PURSUER_POSITION_SUBSCRIBER = rospy.Subscriber("/pursuer/odom", Odometry, callback = get_robot_location, callback_args = {"player_type": "pursuer", "verbose": False})
    EVADER_POSITION_SUBSCRIBER = rospy.Subscriber("/evader/odom", Odometry, callback = get_robot_location, callback_args = {"player_type": "evader", "verbose": False})

    # different cmd publishers to contorl pursuers and evader robots differently
    PURSUER_CMD_PUBLISHER = rospy.Publisher("pursuer/cmd_vel", Twist, latch=True, queue_size=1)
    EVADER_CMD_PUBLISHER = rospy.Publisher("evader/cmd_vel", Twist, latch=True, queue_size=1)
    

    
    
    # create action space
    action_space = []
    # give the agent the ability to reverse, stop, and go forward
    translation_actions= [-0.2, -0.1, 0, 0.1, 0.2]
    rotational_actions = [-60, -40, -20, -1, 0, 1, 20, 40, 60]
    _get_permutations(0, [translation_actions, rotational_actions] ,list(), 2, action_space)

    # create new DQN agent
    pursuer_agent = DQN_Agent(agent_type = "pursuer", input_layer_size = 9, output_layer_size = len(translation_actions)* len(rotational_actions), hidden_layers = [50,50,50,50,50],\
        action_space = action_space, activation_function = relu, activation_function_derivative = relu_derivative, num_steps_to_update_network = 2000, batch_size = 64,
        learning_rate = 0.00025, discount_factor = 0.99, get_agent_lidar_readings = get_lidar_readings, agent_take_action_function = robot_take_action, get_game_information= get_game_information)

    # load in existing DQN agent
    # pursuer_agent = DQN_Agent.load_agent("DQN_pursuer_best_training_80%_ttr_5_pillars_25k.txt")

    # load in random-walking agent
    evader_agent = Random_Walking_Agent("evader", get_lidar_readings, robot_take_action, get_game_information, 0.25)
    
    train("pursuer", pursuer_agent, evader_agent, total_episodes= 100000, starting_epsilon=0.2, max_epsilon=0.95, episode_time_limit=45, time_to_apply_action= 0.5, do_initial_test=False, allow_player_manual_rescue=False)
    # test("pursuer", pursuer_agent, evader_agent, total_episodes=100, episode_time_limit= 90, time_to_apply_action=0.5, allow_evader_manual_rescue= True, allow_pursuer_manual_rescue=False)

if __name__ == "__main__":
    main()