#!/usr/bin/env bash
# This script installs Intera SDK & Robot SDK on Ubuntu 16.04
# This installation is based on page: http://sdk.rethinkrobotics.com/intera/Workstation_Setup
# If you have any problem, please visit above page for more information
# NOTE: Run this file without root permission (do not use 'sudo')

## INSTALL INTERA SDK DEPENDENCIES
# Install SDK Dependencies
sudo apt-get update
sudo apt-get install git-core python-argparse python-wstool python-vcstools python-rosdep ros-kinetic-control-msgs ros-kinetic-joystick-drivers ros-kinetic-xacro ros-kinetic-tf2-ros ros-kinetic-rviz ros-kinetic-cv-bridge ros-kinetic-actionlib ros-kinetic-actionlib-msgs ros-kinetic-dynamic-reconfigure ros-kinetic-trajectory-msgs ros-kinetic-rospy-message-converter

## INSTALL INTERA ROBOT SDK
# Download the SDK on your Workstation
cd ~/ros_ws/src
wstool init .
git clone https://github.com/RethinkRobotics/sawyer_robot.git
wstool merge sawyer_robot/sawyer_robot.rosinstall
wstool update
#Source ROS Setup
source /opt/ros/kinetic/setup.bash
# Build workspace
cd ~/ros_ws
catkin_make
# Copy the intera.sh script
cp ~/ros_ws/src/intera_sdk/intera.sh ~/ros_ws
# Move to ros workspace
cd ~/ros_ws
echo 'Install complete!!!'

