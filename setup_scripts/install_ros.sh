#!/usr/bin/env bash
# This script installs ROS on Ubuntu 16.04
# This installation is based on page: http://sdk.rethinkrobotics.com/intera/Workstation_Setup
# If you have any problem, please visit above page for more information
# NOTE: Run this file without root permission (do not use 'sudo')

## INSTALL ROS
# Setup your sources.list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
# Setup your keys
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# Update to Latest Software Lists
sudo apt-get update
# Install ROS Kinetic Desktop Full
sudo apt-get install ros-kinetic-desktop-full
# Initialize rosdep (rosdep enables you to easily install system dependencies for source you want to compile and is required to run some core components in ROS)
sudo rosdep init
rosdep update
# Install rosinstall
sudo apt-get install python-rosinstall
# Create ROS Workspace (note: `ros_ws` is equivalent to `catkin_workspace` in ROS world)
mkdir -p ~/ros_ws/src
# Source ROS Setup
source /opt/ros/kinetic/setup.bash
# Build workspace
cd ~/ros_ws
catkin_make
echo 'Install complete!!!'

