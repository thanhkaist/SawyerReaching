import sys

sys.path.append('../')
from utils import load_policy, run_policy
from tensorboardX import SummaryWriter
from spinup.utils.run_utils import setup_logger_kwargs
import os

# For Sawyer multiworld
from multiworld.envs.mujoco import register_mujoco_envs as register_goal_example_envs
from multiworld.core.flat_goal_env import FlatGoalEnv

from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv


register_goal_example_envs()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, default='./logs/ddpg_test_SawyerReachXYZEnv_multi_real_2/ddpg/ddpg_s0/')
    parser.add_argument('--len', '-l', type=int, default=50)
    parser.add_argument('--episodes', '-n', type=int, default=50)
    parser.add_argument('--render', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1, help='Choose iter want to run in saved folder')
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs/ddpg_test')
    parser.add_argument('--exp_name', type=str, default='evaluate')
    parser.add_argument('--env', type=str, default='SawyerReachXYZenv_multi')
    args = parser.parse_args()

    env, get_action = load_policy(args.saved_model,
                                  args.itr if args.itr >= 0 else 'last',
                                  args.deterministic)
    tensor_board = None
    logdir_ext = os.path.join(args.logdir + '_' + args.env + '_evaluate')
    if not os.path.exists(logdir_ext):
        os.mkdir(logdir_ext)

    if args.use_tensorboard:
        tensor_board = SummaryWriter(logdir_ext)

    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, data_dir=logdir_ext)
    print(logger_kwargs)
    run_policy(env, get_action, args.len, args.episodes, args.render, tensor_board)