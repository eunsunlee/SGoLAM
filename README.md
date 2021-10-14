# SGoLAM - Simultaneous Goal Localization and Mapping

PyTorch implementation of the MultiON runner-up entry, SGoLAM: Simultaneous Goal Localization and Mapping [[Talk Video].](https://www.youtube.com/watch?v=XgT2w6rUwjM) Our method does not employ any training of neural networks, but shows competent performance in the [MultiON benchmark.](https://eval.ai/web/challenges/challenge-page/805/leaderboard/2202) In fact, we outperform the winning entry by a large margin in terms of success rate. 

![alt text](https://github.com/eunsunlee/SGoLAM/blob/main/rank.png)

We encourage future participants of the MultiON challenge to use our code as a starting point for implementing more sophisticated navigation agents. If you have any questions on running SGoLAM please leave an issue.

## Notes on Installation
To run experiments locally/on a server, follow the 'bag of tricks' below:
1. Please abide by the steps provided in the [original MultiON repository.](https://github.com/saimwani/multiON) (Don't bother looking at other repositories!)
2. Along the installation process, numerous dependency errors will occur. Don't look for other workarounds and just humbly install what is missing.
3. For installing Pytorch and other CUDA dependencies, it seems like the following command works: `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch`.
4. By the way, `habitat-lab` installation is much easier than `habitat-sim`. You don't necessarily need to follow the instructions provided in the MultiON repository for `habitat-lab`. Just go directly to the [habitat-lab repository](https://github.com/facebookresearch/habitat-lab) and install habitat-lab. However, for `habitat-sim`, you must follow MultiON's directions; or a pile of bugs will occur.
5. One `python evaluate.py` is run, a horrifying pile of dependency errors will occur. Now we will go over some of the prominent ones.
6. To solve `AttributeError: module 'attr' has no attribute 's'`, run `pip uninstall attr` and then run `pip install attrs`.
7. To solve `ModuleNotFoundError: No module named 'imageio'`, run `pip install imageio-ffmpeg`.
8. To solve `ImportError: ModuleNotFoundError: No module named 'magnum'`, run `pip install build/deps/magnum-bindings/src/python`.
9. The last and most important 'trick' is to google errors. The Habitat team seems to be doing a great job answering GitHub issues. Probably someone has already ran into the error you are facing.
10. If additional 'tricks' are found, feel free to share by appending to the list starting from **here**.
`
## Docker Sanity Check (Last Modified: 2021.03.26:20:11)
A number of commands to take for docker sanity check.
### Login
First, login to the dockerhub repository. As our accounts don't support private repositories with multiple collaborators, we need to share a single ID.
For the time being let's use my ID. 
Type the following command
```
docker login
```
Now one will be prompted a user ID and PW. Please type ID: esteshills PW: 82magnolia.

### Pull Image
I have already built an image ready for preliminary submission. It can be easily pulled using the following command.
```
docker pull esteshills/multion_test:tagname
```

### Run Evaluation
To make an evaluation for standard submission, run the following command. Make sure `DATA_DIR` and `ORIG_DATA_DIR` from `scripts/test_docker.sh` are modified before running.
```
cd scripts/
./test_docker.sh
```

### Playing around with Docker Images
One may want to further examine the docker image. Run the following command.
```
cd scripts/
./test_docker_bash.sh
```
Again, make sure `DATA_DIR` and `ORIG_DATA_DIR` from `scripts/test_docker.sh` are modified before running.
Note that the commands provided in the [MultiON repository](https://github.com/saimwani/multiON) can be run inside the container.
For example:
```
python habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_multinav_no_map.yaml --agent-type no-map --run-type eval
```
In order to run other baselines, i) modify the checkpoint path in the .yaml file, ii) download the model checkpoint, iii) change the agent type.

## Preventing Hassles with Docker (Last Modified: 2021.04.08:09:07)
Now we probably don't need to develop with docker. Just plug in your favorite agent following the instructions provided below.

### Plug-and-Play New Agents
One can easily test new agents by providing the file name containing agent implementation.
To implement a new agent, please refer to `agents/example.py`.
To test a new agent and get evaluation results, run the following command (this is an example for the `no_map` baseline).
```
python evaluate.py --agent_module no_map_walker --exp_config habitat_baselines/config/multinav/ppo_multinav_no_map.yaml --checkpoint_path model_checkpoints/ckpt.0.pth --no_fill
```
In addition, one can change the number of episodes to be tested.
However, this feature is only available in the `annotated` branch, as it requires a slight modification in the core `habitat` repository.
Run the following command to change the number of episodes.
While it will not produce any bugs in the `main` branch as well, the argument will have no effect.
```
python evaluate.py --agent_module no_map_walker --exp_config habitat_baselines/config/multinav/ppo_multinav_no_map.yaml --checkpoint_path model_checkpoints/ckpt.0.pth --no_fill --num_episodes 100
```

### Plug-and-Play New Agents from Local Host
#### Running Agents
Suppose one has some implementations of navigation agents that are not yet pushed to `agents/`.
These could be tested on-the-fly using a handy script provided in `scripts`.
First, put all the agent implementations inside `extern_agents/`, similar to implementations in `agents/`.
Then run the following command with the agent module you are trying to run, for example if the new agent module is located in `extern_agents/new_agent.py`, run
```
./scripts/test_docker_agent.sh new_agent
```
Make sure the agents are located in the `extern_agents/` folder.
This way, there is no need to directly hassle with docker; docker is merely used as a black box for running evaluations.

Now suppose one needs to debug the agent in the docker environment.
This could be done by running the following script; it will open bash with `extern_agents/` mounted.
```
./scripts/test_docker_agent_bash.sh
```

To run evaluations **inside** the docker container, run the following command with the agent module name (in this case `new_agent`) provided.
```
./scripts/extern_eval.sh new_agent
```

#### Playing Agent Episodes with Video
Agent trajectories per episode can be visualized with the scripts in `scripts/`.
Again, put all the agent implementations inside `extern_agents/`.
Then run the following command with the agent module you are trying to run, for example if the new agent module is located in `extern_agents/new_agent.py`, run
```
./scripts/test_docker_agent_video.sh new_agent 
```
Make sure the mount paths are set correctly inside `./scripts/test_docker_agent_video.sh`.

To run evaluations **inside** the docker container, run the following command with the agent module name (in this case `new_agent`) and video save directory (in this case `./test_dir`) provided.
```
./scripts/extern_eval_video.sh new_agent ./test_dir
```

#### Caveats
The original implementations assume two GPUs to be given. Therefore bugs may occur if only a single GPU is present. In this case do **not** run the docker scripts directly, as it will return errors. Instead, connect to a docker container with bash and first modify the baseline `.yaml` configuration so that it only uses a single GPU. Then, run the `*_eval*.sh` scripts. I am planning on remedying this issue with a similar plug-and-play fashion, but for the time being, stick to this procedure.
