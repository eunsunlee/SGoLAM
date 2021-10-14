def fill_args(args):
    """
    Fill additional args for submission.
    """
    args.agent_module = 'dstar_sgolam_walker'
    args.checkpoint_path = None
    args.exp_config = 'configs/baselines/dstar_proto_sgolam.yaml'
    args.num_episodes = 25
    
    return args
