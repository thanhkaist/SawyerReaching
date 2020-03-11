# Sawyer Reaching 

## Install dependency 
1. ros_kinetic + intera_interface_sdk \
https://sdk.rethinkrobotics.com/intera/Workstation_Setup
2. gazebo \
https://sdk.rethinkrobotics.com/intera/Gazebo_Tutorial
3. install sawyer_control \
Follow this[sawyer_control/README.md]
Change this step 
```
run `git clone https://github.com/mdalal2020/sawyer_control.git` in ~/catkin_ws/src/
```
to
```
ln -s (SawyerReachingpath)/sawyer_control .
```
Change this step
```
pip install -r system_python_requirements.txt
```
to
```
pip install -r system_python_requirements.txt --no-deps
```

Note: you need to install sawyer_control to env with 
```
pip install -e .
```
and run $python_path alias before you can use that package
4. install multiworld
Install multiworld package into your virtual env that you have created in step 3.
```
cd (SawyerReachingpath)/sawyer_control
pip install -e .
```

## Make a ros workspace 
```
cd ~
mkdir rl_ws/src
cd rl_ws
catkin_make
```

## 


## Working on real robot

Step 1: 
Set up parameters for the hand-eye camera
```
actenv
saw2
rosrun intera_examples camera_display.py -c right_hand_camera -x 10 -g 10
```

Step 2:
Launch environment
```
roslaunch sawyer_reach main.launch
```

Step 3:
```
cd ~/SawyerReaching/algorithm/ddpg
python  grasp_server.py
```
--> run subscriber

Step 4:
```
actenv
saw2
exp_nodes
source activate thanh
export PYTHONPATH=~/anaconda2/envs/thanh/lib/python3.6/site-packages:/usr/lib/python3/dist-packages:~/hieu_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
cd ~/thanh_code/sawyer/ddpg
python object_grasping.py
```

```
(thanh) [intera - http://021709CP00082.local:11311] hsbk@HSBK:~/thanh_code/sawyer/ddpg$ python markemultiworld_multi_test_policy.py
```

To create environment, check requirements.txt file

Dependencies:

- sawyer_control : configs are changed
- multiworld
- sawyer_thanh: for get ar pose marker
- spinningup
- sawyer folder: source code for training and testing