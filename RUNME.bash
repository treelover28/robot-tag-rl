#! /bin/bash

cd src;
chmod +x robot_tag.py;
cd ..;
export TURTLEBOT3_TAGGER="waffle";
export TURTLEBOT3_TAGGEE="burger";
source ../../devel/setup.bash;
gnome-terminal -- roslaunch robot_tag robottag.launch;
sleep 2;
gnome-terminal -- rosrun robot_tag robot_tag.py;
sleep 2;
gnome-terminal -- rosrun tf2_tools view_frames.py && evince frames.pdf 
