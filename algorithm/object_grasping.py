"""

@author: aim lab
@email: cd_yoo@kaist.ac.kr
"""

import os
import argparse
import numpy as np

from utils import load_policy, run_policy

from multiworld.envs.mujoco import register_mujoco_envs as register_goal_example_envs
from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

register_goal_example_envs()

import rospy
from sawyer_thanh.srv import target, targetResponse, targetRequest
from sawyer_control.srv import grasping
from sawyer_control.srv import angle_action


class ObjectGrasping:
    """
    This class provides the utilities to grasp object including:
        - Find object's position based on /ar_pose_marker service
        - Move joints to specific angles
        - Move end-effector to specific position by using policy learned by RL algorithm
    """

    def __init__(self, use_hand_cam=False):
        self.use_hand_cam = use_hand_cam

        # Transformation matrix from camera's frame -> based frame
        # self.TRANSFORMATION_MATRIX = np.array([[0.11491126, 0.88002959, -0.46080724, 1.0704251219017176],
        #                                        [0.99326509, -0.0948642, 0.06652247, 0.02981537521689703],
        #                                        [0.01482763, -0.46534793, -0.88500364, 0.6268248987975156],
        #                                        [0., 0., 0., 1.]])
        self.TRANSFORMATION_MATRIX = np.array([[-0.15316623, 0.86485568, -0.47808446, 1.06231099],
                                               [0.97058596, 0.22259649, 0.09172615, -0.08591922],
                                               [0.18574981, -0.44997272, -0.87351105, 0.62519807],
                                               [0., 0., 0., 1.]])

        self.angle_defaul_cam = [-0.9347021484375, -0.066611328125, -2.09948828125,
                                 -2.4536884765625, -1.90233984375, -2.909759765625, -2.622689453125]
        self.angle_init_for_grasp = [0.51219827, -0.35472363, -0.69057131, 1.43175006, -2.19978213,
                                     -0.83249319, -1.90052831]
        self.angle_for_place_object = [-0.34514549, 0.24693164, -1.2170068, 1.22242475, 1.65923345,
                                       1.15603614, 0.06596191]
        self.msg_close = True
        self.msg_open = False

        env = SawyerReachXYZEnv(
            action_mode='position',
            position_action_scale=0.1,
            config_name='austri_config',
            reset_free=False,
            max_speed=0.05,
            fix_goal=True,
        )
        self.env = FlatGoalEnv(env, append_goal_to_obs=True)

        os.system('clear')
        print('[AIM-INFO] Initializing robotic grasping...')
        for _ in range(5):
            self.move_to_angle(angle=self.angle_init_for_grasp, duration=2)
        print('[AIM-INFO] Initialize done.')

    def go_to_camera_view_position(self):
        duration = 2
        self.move_to_angle(self.angle_defaul_cam, duration)

    def go_to_place_position(self):
        duration = 7
        self.move_to_angle(self.angle_for_place_object, duration)

    def move_to_angle(self, angle, duration):
        rospy.wait_for_service('angle_action')
        try:
            execute_action = rospy.ServiceProxy('angle_action', angle_action, persistent=True)
            execute_action(angle, duration)
            return None
        except rospy.ServiceException as e:
            print('[AIM-ERROR] Error when moving to angle: ', angle)

    def locate_object(self):
        service_name = "/locate_object"
        service = rospy.ServiceProxy(service_name, target)
        service.wait_for_service()
        print("[AIM-INFO] Connect to service {} successfully.".format(service_name))

        while True:
            req = targetRequest()
            req.data = 0
            resp = service.call(req)
            if resp.pose is not ():
                print('[AIM-INFO] Object detected')
                break
            elif self.use_hand_cam:
                print('[AIM-INFO] Cannot detect object...')
                self.go_to_camera_view_position()
            else:
                print('[AIM-INFO] Cannot detect object...')

        return resp.pose

    def get_object_location(self):
        obj_pos_cam_frame = self.locate_object()  # w.r.t. camera frame

        print("[AIM-DEBUG] Object in camera frame: (%.4f, %.4f, %.4f)" % (
            obj_pos_cam_frame[0], obj_pos_cam_frame[1], obj_pos_cam_frame[2]))
        if self.use_hand_cam:
            obj_pos_based_frame = list(obj_pos_cam_frame)
        else:
            obj_pos_homo = np.hstack([obj_pos_cam_frame, 1])
            obj_pos_based_frame = np.matmul(self.TRANSFORMATION_MATRIX, obj_pos_homo)

        print("[AIM-DEBUG] Object in based frame: (%.4f, %.4f, %.4f)" % (
            obj_pos_based_frame[0], obj_pos_based_frame[1], obj_pos_based_frame[2]))
        obj_pos_based_frame[2] = obj_pos_based_frame[2] + 0.15
        print("[AIM-DEBUG] Object in based frame with offset: (%.4f, %.4f, %.4f)" % (
            obj_pos_based_frame[0], obj_pos_based_frame[1], obj_pos_based_frame[2]))

        return list(obj_pos_based_frame[:3])

    def request_grasp(self, data):
        rospy.wait_for_service('grasping')
        execute_action = rospy.ServiceProxy('grasping', grasping, persistent=True)
        execute_action(data)

    def move_to_pos(self, goal):
        self.env.wrapped_env._state_goal = np.array(goal)
        print('[AIM-INFO] Moving to reset position...')
        for _ in range(5):
            self.env.reset()
        print('[AIM-INFO] Starting move to target position...')
        run_policy(self.env, get_action, 15, 1, False)


def main():
    grasp = ObjectGrasping(use_hand_cam=True)

    grasp.request_grasp(grasp.msg_open)  # Open gripper
    obj_pos = grasp.get_object_location()

    grasp.move_to_pos(obj_pos)

    grasp.request_grasp(grasp.msg_close)

    grasp.go_to_place_position()

    grasp.request_grasp(grasp.msg_open)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str,
                        default='/home/tung/workspace/ee331_project/thanh_code/sawyer/ddpg/logs/ddpg_test_SawyerReachXYZEnv_multi_real_2/ddpg/ddpg_s0/')
    parser.add_argument('--len', '-l', type=int, default=60)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--logdir', type=str,
                        default='/home/tung/workspace/ee331_project/thanh_code/sawyer/ddpg/logs/ddpg_test')
    parser.add_argument('--exp_name', type=str, default='evaluate')
    parser.add_argument('--env', type=str, default='SawyerReachXYZenv_multi')
    args = parser.parse_args()

    # global env
    global get_action
    _, get_action = load_policy(args.saved_model, 'last', args.deterministic)

    main()
