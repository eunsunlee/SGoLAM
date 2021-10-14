# DATA_DIR is where the 'data' folder resides within the multion-challenge folder
# If you used symlink, specify the original data directory in ORIG_DATA_DIR
DATA_DIR=/home/jhkim/Projects/multion-challenge/data/
ORIG_DATA_DIR=/home/jhkim/Datasets/mp3d_habitat/mp3d/

# Full Docker command if symlink is used
docker run --rm --ipc="host" -v $DATA_DIR:/multion-chal-starter/data/ -v $ORIG_DATA_DIR:/home/jhkim/Datasets/mp3d_habitat/mp3d/ --runtime=nvidia esteshills/multion_test:tagname

# Full Docker command if symlink is not used: comment the line above and use this one instead
# docker run --rm --ipc="host" -v $DATA_DIR:/multion-chal-starter/data/ --runtime=nvidia esteshills/multion_test:tagname

