import sys

sys.path.append('../')
from utils import load_policy, run_policy
from tensorboardX import SummaryWriter
from util.run_utils import setup_logger_kwargs
import os

# For Sawyer
from multiworld.envs.mujoco import register_reaching_envs as register_reaching_envs
register_reaching_envs()
from multiworld.core.flat_goal_env import FlatGoalEnv



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, default='./logs/ddpg_test_SawyerPushAndReachArenaEnv-v0/ddpg/ddpg_s0/')
    parser.add_argument('--len', '-l', type=int, default=50)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--render', '-nr', action='store_false')
    parser.add_argument('--itr', '-i', type=int, default=-1, help='Choose iter want to run in saved folder')
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs/ddpg_test')
    parser.add_argument('--exp_name', type=str, default='evaluate')
    parser.add_argument('--env', type=str, default='SawyerReachXYEnv-v1')
    args = parser.parse_args()

    _, get_action = load_policy(args.saved_model,
                                  args.itr if args.itr >= 0 else 'last',
                                  args.deterministic)
    tensor_board = None
    env = SawyerReachXYZEnv(
            action_mode='position',
            position_action_scale=0.1,
            config_name='austri_config',
            reset_free=False,
            max_speed=0.05,
            fix_goal=False,
            fixed_goal=(0.53,0.0,0.15)
        )

    env = FlatGoalEnv(env, append_goal_to_obs=True)
    env.reset()
    logdir_ext = os.path.join(args.logdir + '_' + args.env + '_evaluate')
    if not os.path.exists(logdir_ext):
        os.mkdir(logdir_ext)

    if args.use_tensorboard:
        tensor_board = SummaryWriter(logdir_ext)

    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, data_dir=logdir_ext)
    run_policy(env, get_action, args.len, args.episodes, args.render, tensor_board)