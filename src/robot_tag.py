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

DISTANCE_FROM_OBSTACLE = 0.25
TAGGER_STATE_DISCRETIZED = None 
TAGGER_STATE_CONTINUOUS = None
TAGGER_POSITION = None 

TAGGEE_STATE_DISCRETIZED = None 
TAGGEE_STATE_CONTINUOUS = None
TAGGEE_STUCK = False

# Gameplay hyperparameters
TAGGER_STUCK = False
TAGGEE_POSITION = None 
TIMEOUT = False
GAME_TIME = 30 # a round/traning episode last maximum 30 seconds

ROTATIONAL_ACTIONS = [45,20,0,-20,-45]
TRANSLATION_SPEED = 0.2 
DIRECTIONAL_STATES = ["Front", "Left", "Right", "Opponent Position"]
FRONT_RATINGS = ["Close", "OK", "Far"]
LEFT_RATINGS = ["Too Close", "Close", "OK", "Far", "Too Far"]
RIGHT_RATINGS = ["Too Close", "Close", "OK", "Far", "Too Far"]
OPPONENT_POSITION = ["Close Left", "Left", "Close Front", "Front", "Right", "Close Right", "Bottom", "Close Bottom"]


