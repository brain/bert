#
# common
# NOTE: not to be use independtly
#

HIDE ?= @
SHELL := /bin/bash
NETWORK := brain-dev-network

network:
	-$(HIDE)docker network create --attachable -d bridge $(NETWORK) > /dev/null 2>&1 || true

.deps: # autogenerate .env for compose  with variables from Makefile
	$(HIDE)echo "$(foreach V,$(sort $(.VARIABLES)),$(if $(filter-out environment% default automatic,$(origin $V)),$V='$($V)'))"|xargs -n1 > .env
	$(HIDE)$(MAKE) network

dockerize:
	-$(HIDE)docker rmi $(REGISTRY)/brain/dockerize
	$(HIDE)docker run --rm -it -v $(PWD):/brain/src -ePWDIR=$(PWD) $(REGISTRY)/brain/dockerize base

pyclear:
	$(HIDE)find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

jupyter:
	$(HIDE)docker run -it --rm $(VOLUME) $(DOCKER_IMAGE) pip install --user jupyter --upgrade
	$(HIDE)docker run -it --rm -p8888 --network $(NETWORK) $(VOLUME) $(DOCKER_IMAGE) jupyter notebook --no-browser --ip 0.0.0.0 --port=8888 --allow-root
