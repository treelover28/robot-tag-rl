#! /bin/bash

export TURTLEBOT3_TAGGER="waffle";
export TURTLEBOT3_TAGGEE="waffle_pi";
source ../../devel/setup.bash;
gnome-terminal -- roslaunch robot_tag robottag.launch;


