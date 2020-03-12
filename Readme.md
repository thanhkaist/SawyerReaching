# Sawyer Reaching 

## Update submodule
```
git submodule init
git submodule update
```
## Install dependency 
1. ros_kinetic + intera_interface_sdk \
https://sdk.rethinkrobotics.com/intera/Workstation_Setup
2. gazebo \
https://sdk.rethinkrobotics.com/intera/Gazebo_Tutorial
3. install sawyer_control \
Follow this [sawyer_control/README.md] \
Remember: \

* Change this step 
```
run `git clone https://github.com/mdalal2020/sawyer_control.git` in ~/catkin_ws/src/
```
to
```
cd (path to ros_ws)/src
ln -s (path to SawyerReaching)/sawyer_control .
```
* Change this step
```
pip install -r system_python_requirements.txt
```
to
```
pip install -r system_python_requirements.txt --no-deps
```

Note: you also need to install sawyer_control to your virtual env (e.g my_env) with 
```
cd (path to SawyerReaching)/sawyer_control
pip install -e .
```
and run $python_path alias before you can use that package 

4. install multiworld 

Install multiworld package into your virtual env (e.g my_env) that you have created in step 3.
```
cd (path to SawyerReaching)/multiworld
pip install -e .
```

5. install algorithm dependency 

```
cd (path to SawyerReaching)
pip install -r requirement
```


## Training algorithm 
### Working with simulation

Useful alias 
```
alias ros_enable="source /opt/ros/kinetic/setup.bash; source ~/rl_ws/devel/setup.bash"
alias saw_sim="ros_enable; ./intera.sh sim"
alias saw="cd ~/ros_ws/; ./intera.sh; cd ~/"
alias exp_nodes="roslaunch ~/ros_ws/src/sawyer_control/exp_nodes.launch"
alias python_path="export PYTHONPATH=/home/$your_user$/miniconda2/envs/rl_ros/lib/python3.5/site-packages:$PYTHONPATH:/opt/ros/kinetic/lib/python2.7/dist-packages/"
```
Run sawyer gazebo
```
saw_sim
roslaunch sawyer_gazebo sawyer_world
```
Run control nodes (in another terminal)
```
saw_sim
exp_node
```
Run visualization nodes if needed (in another terminal)
```
saw_sim
roslauch ~/ros_ws/src/sawyer_control/main.launch
```
In rviz, File->Open Config-> (path to SawyerReaching)/reachingXYZ.rviz

Train your algorithm (in another terminal)
```
saw_sim
conda activate my_env
python_path
cd (path to SawyerReaching)/algorithm/ddpg
python ddpg --env $name$

```
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