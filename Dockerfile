FROM multi_on:latest
ADD evaluate.py /multion-chal-starter
ADD submit.sh /multion-chal-starter
ADD submit_args.py /multion-chal-starter
ADD utils/ /multion-chal-starter/utils
ADD agents/ /multion-chal-starter/agents
ADD configs/ /multion-chal-starter/configs
WORKDIR /multion-chal-starter
CMD ["/bin/bash", "-c", "source activate habitat && bash submit.sh"]
