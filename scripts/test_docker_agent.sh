# Script for directly running an external agent

# DATA_DIR is where the 'data' folder resides within the multion-challenge folder
# If you used symlink, specify the original data directory in ORIG_DATA_DIR
# AGENT_DIR is the directory containing agent implementation
DATA_DIR=/home/jhkim/Projects/multion-challenge/data/
ORIG_DATA_DIR=/home/jhkim/Datasets/mp3d_habitat/mp3d/
AGENT_DIR=/home/jhkim/Projects/multi_on_win/extern_agents/
AGENT_MODULE="$1"

# Full Docker command if symlink is used
docker run -it --rm --ipc="host" -v $DATA_DIR:/multion-chal-starter/data/ -v $ORIG_DATA_DIR:/home/jhkim/Datasets/mp3d_habitat/mp3d/ -v $AGENT_DIR:/multion-chal-starter/extern_agents/ --runtime=nvidia esteshills/multion_test:tagname ./scripts/extern_eval.sh $AGENT_MODULE

# Full Docker command if symlink is not used: comment the line above and use this one instead
# docker run -it --rm --ipc="host" -v $DATA_DIR:/multion-chal-starter/data/ -v $AGENT_DIR:/multion-chal-starter/extern_agents/ --runtime=nvidia esteshills/multion_test:tagname ./scripts/extern_eval.sh $AGENT_MODULE  

