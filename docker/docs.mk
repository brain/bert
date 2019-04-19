#
# common
# NOTE: not to be used independtly
#

HIDE ?= @
SHELL := /bin/bash
RENDER_DOCS ?= false

docs-push: # documentation at docs.braininc.net/
	$(HIDE)docker run -d --name $(DOCKER_CONTAINER)-docs $(REGISTRY)/brain/docs:slate
	$(HIDE)docker cp ${PWD}/docs/index.html.md $(DOCKER_CONTAINER)-docs:/brain/src/source/includes/$(DOCKER_CONTAINER).html.md
	$(HIDE)docker container commit $(DOCKER_CONTAINER)-docs $(REGISTRY)/brain/docs:slate
	$(HIDE)docker push $(REGISTRY)/brain/docs:slate
	$(HIDE)docker rm -f $(DOCKER_CONTAINER)-docs

docs:
	$(HIDE)echo 'docker run -it --rm $(VOLUME) $(DOCKER_IMAGE) apidoc -i docs/apidocs.yml -o docs/index.html'
