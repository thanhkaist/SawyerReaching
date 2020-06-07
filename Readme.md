# Sawyer Reaching 

## Install sourcecode
```
mkdir ~/ee331 && cd ~/ee331
git clone https://github.com/thanhkaist/SawyerReaching.git
cd SawyerReaching
```

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
Remember: 
* Use your (path to ros_ws)/src instead of ~/catkin_ws/src \
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
* No need to install kinect2 bridge (step 14) \
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

## Training algorithm 
### Requirement 

Install algorithm dependency
```
conda activate my_env
pip install -r algorithm/requirement.txt
```

Add useful alias to ~/.bashrc
```
alias ros_enable="source /opt/ros/kinetic/setup.bash; source ~/$ros_ws$/devel/setup.bash"
alias saw_sim="ros_enable; ./intera.sh sim"
alias saw="cd ~/ros_ws/; ros_enable; ./intera.sh "
alias exp_nodes="roslaunch ~/ros_ws/src/sawyer_control/exp_nodes.launch"
alias python_path="export PYTHONPATH=/home/$your_user$/miniconda2/envs/$ros_ws$/lib/python3.5/site-packages:$PYTHONPATH:/opt/ros/kinetic/lib/python2.7/dist-packages/"
```

### Working with simulation

Run sawyer gazebo
```
saw_sim
roslaunch sawyer_gazebo sawyer_world.launch
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

Test your algorithm

```
saw_sim
conda activate my_env
python_path
cd (path to SawyerReaching)/algorithm/ddpg
python ddpg_test 

```

### Working on real robot
 
Set up parameters for the hand-eye camera
```
saw
rosrun intera_examples camera_display.py -c right_hand_camera -x 10 -g 10
```

Run control nodes (in another terminal)
```
saw
exp_node
```

Run visualization nodes and vision node (in another terminal)
```
saw
roslauch ~/ros_ws/src/sawyer_control/main.launch
```

Run grasping (in another terminal)
```
saw
conda activate my_env
python_path
cd (path to SawyerReaching)/algorithm/ddpg
python  grasp_server.py
```

