# Agents
Here we will keep agents that will be tested until the challenge.
The current list of agents implemented are as follows:
- `example.py` : Sample agent with random policies.
- `no_map_walker.py`: No map baseline using solely GRU memory.
- `learned_map_walker.py`: Projected neural map & object recognition map baseline. 
  Their implementations are identical. One can choose between the two by specifying the configuration `.yaml` file.

One can run a new agent by providing its module name to `evaluate.py`.
Please refer to README.md at the root folder for more details.
