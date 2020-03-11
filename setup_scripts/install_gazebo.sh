#!/usr/bin/env bash
# This script installs Sawyer Simulator (Gazebo) on Ubuntu 16.04
# This installation is based on page: http://sdk.rethinkrobotics.com/intera/Gazebo_Tutorial
# If you have any problem, please visit above page for more information
# NOTE: Run this file without root permission (do not use 'sudo')

sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install gazebo7
sudo apt-get install libgazebo7-dev

# Installing the following software packages
sudo apt-get install gazebo7 ros-kinetic-qt-build ros-kinetic-gazebo-ros-control ros-kinetic-gazebo-ros-pkgs ros-kinetic-ros-control ros-kinetic-control-toolbox ros-kinetic-realtime-tools ros-kinetic-ros-controllers ros-kinetic-xacro python-wstool ros-kinetic-tf-conversions ros-kinetic-kdl-parser ros-kinetic-sns-ik-lib

# Install sawyer_simulator
cd ~/ros_ws/src
git clone https://github.com/RethinkRobotics/sawyer_simulator.git
cd ~/ros_ws/src
wstool init .
wstool merge sawyer_simulator/sawyer_simulator.rosinstall
wstool update
# Build Source
source /opt/ros/kinetic/setup.bash
cd ~/ros_ws
catkin_make
echo 'Install complete!!!'

