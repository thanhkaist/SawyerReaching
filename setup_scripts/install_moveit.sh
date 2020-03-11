#!/usr/bin/env bash
# This script installs MoveIt on Ubuntu 16.04
# This installation is based on page: http://sdk.rethinkrobotics.com/intera/MoveIt_Tutorial
# If you have any problem, please visit above page for more information
# NOTE: Run this file without root permission (do not use 'sudo')

# Make sure to update your sources
sudo apt-get update
# Install MoveIt!
sudo apt-get install ros-kinetic-moveit
# Installing and building sawyer MoveIt Repo
cd ~/ros_ws
./intera sim
cd ~/ros_ws/src
wstool merge https://raw.githubusercontent.com/RethinkRobotics/sawyer_moveit/master/sawyer_moveit.rosinstall
wstool update
cd ~/ros_ws
catkin_make
echo 'Install complete!!!'

