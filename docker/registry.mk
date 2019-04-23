#
# common
# NOTE: not to be use independtly
#

HIDE ?= @
SHELL := /bin/bash
REGISTRY ?= docker.brainllc.net
BRANCH_NAME ?= `git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/'`

push:
	$(HIDE)docker tag $(DOCKER_IMAGE) $(REGISTRY)/$(DOCKER_IMAGE):$(BRANCH_NAME)
	$(HIDE)docker push $(REGISTRY)/$(DOCKER_IMAGE):$(BRANCH_NAME)
	$(HIDE)docker rmi $(REGISTRY)/$(DOCKER_IMAGE):$(BRANCH_NAME)
	-$(HIDE)docker rmi $(DOCKER_IMAGE)

pull:
	$(HIDE)docker pull $(REGISTRY)/$(DOCKER_IMAGE):$(BRANCH_NAME)
	$(HIDE)docker tag  $(REGISTRY)/$(DOCKER_IMAGE):$(BRANCH_NAME) $(DOCKER_IMAGE)

login:
	$(HIDE)docker login --username $(DOCKER_USERNAME) --password $(DOCKER_PASSWORD) $(REGISTRY)

restart:
	$(HIDE)docker restart $(DOCKER_CONTAINER)

stop:
	$(HIDE)$(MAKE) stop.$(DOCKER_CONTAINER)

stop.%:
	$(HIDE)docker rm -f $*

enter.%:
	$(HIDE)docker exec -it $* sh -c 'test -e /bin/bash && /bin/bash || sh'

enter:
	$(HIDE)$(MAKE) enter.$(DOCKER_CONTAINER)

clean-all:
	-$(HIDE)docker ps -aq | xargs docker stop
	-$(HIDE)docker ps -aq | xargs docker rm
	-$(HIDE)docker images -q| xargs docker rmi

clean-containers:
	-$(HIDE)docker ps -aq | xargs docker stop
	-$(HIDE)docker ps -aq | xargs docker rm

clean:
	-$(HIDE)docker rmi $(docker images |grep --color=always none|xargs -n1|grep -A1 none|grep -Ev 'none|--')

ctop:
	$(HIDE)docker run --rm -ti --name=ctop -v /var/run/docker.sock:/var/run/docker.sock quay.io/vektorlab/ctop:latest
