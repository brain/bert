# Siamese BERT for sentence similarity calculation

## How to use
### Installing dependencies
Later, we will make this setup through docker and the `make` commands. For now, follow these steps:
1. Set up python dependencies with these commands
```
cd ~; sudo apt install virtualenv
virtualenv -p python3 deployment-env
source ~/deployment-env/bin/activate
sudo apt-get install mysql-server
sudo apt-get install libmysqlclient-dev
cd <bert repo>; pip3 install -r requirements.txt
pip install tensorflow-gpu==1.13.1
```
(We have to run the last step manually because including tensorflow-gpu in the requirements file causes it to fail on CircleCi.)
If NVIDIA-related dependencies are already set up, you can skip the remaining steps.
2. Install NVIDIA package repos:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
```
3. Install the NVIDIA driver:
```
sudo apt-get install --no-install-recommends nvidia-driver-410
```
4. Reboot your machine
5. Install CUDA and cuDNN:
```
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.4.1.5-1+cuda10.0  \
    libcudnn7-dev=7.4.1.5-1+cuda10.0
```
### Starting the service
In the repo base directory, with the virtualenv started, run:
```
./manage.py run_development_server
```
At the moment, this doesn't work with `run_production_server` because the workers can't find the GPU. We're working to fix this. 
### Calling the API
Here's an example:
```
curl -H 'content-type: application/json' -X POST -d '{"doc1": "lets see captain marvel", "doc2": ["Purchase 3 tickets for Dunkirk via MovieTickets.com.", "Buy the family tickets to Hamilton in San Diego using Vividseats", "Play classical music on Spotify."], "sort": true}' http://localhost:7000/similarities
```
