#!/bin/bash
# Provide the agent name; it should located inside extern_agents.
# It is recommended to import agent configs and load pretrained weights directly inside the agent_module.
# Remember that pretrained weights and configs should also be mounted. One solution would be to put all of them inside extern_agents. This will make things easier.
AGENT_NAME="$1"

source activate habitat && python evaluate.py --external --agent_module $AGENT_NAME --no_fill
