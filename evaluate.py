import agents
import argparse
from habitat.core.challenge import Challenge
import importlib
from submit_args import fill_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", type=str, required=False, choices=["dev", "standard", "challenge", "video"]
    )
    parser.add_argument(
        "--agent_module",
        required=False,
        type=str,
        default="example",
        help="agent module name",
    )
    parser.add_argument(
        "--exp_config", type=str, required=False, help="Config within habitat baselines"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=False, help="Path to checkpoint"
    )
    parser.add_argument(
        '--num_episodes', type=int, required=False, default=None ,help="Number of episodes to evaluate. Only works in dev mode."
    )
    parser.add_argument(
        "--no_fill", action='store_true', required=False, help="If Set, skips fill_args"
    )
    parser.add_argument(
        "--external", action='store_true', required=False, help="If Set, agents are loaded from extern_agents folder"
    )
    parser.add_argument(
        "--video_dir", type=str, default=None, help="Path where videos will be logged"
    )

    args = parser.parse_args()

    if not args.no_fill:
        args = fill_args(args)

    phase = args.phase
    challenge = Challenge(phase = phase)

    if args.phase is None or args.phase == "dev" or args.phase == "video":
        if args.num_episodes is not None:
            challenge.num_episodes = args.num_episodes
    
    if args.external:
        walker = importlib.import_module(f'extern_agents.{args.agent_module}')
    else:
        walker = importlib.import_module(f'agents.{args.agent_module}')
    agent = walker.get_agent(args.exp_config, challenge, args.checkpoint_path)

    if args.video_dir is not None:
        from agents.video_walker import VideoWalker
        args.phase = "video"
        agent = VideoWalker(agent, args.video_dir)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
