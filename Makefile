# make image
# make start

SHELL := /bin/bash
HIDE ?= @
DOCKER_IMAGE ?= brain/bert
DOCKER_CONTAINER ?= bert
VOLUME ?=-v $(PWD):/brain/src
DOCKER_ENV ?=--rm -it

-include ./docker/registry.mk
-include ./docker/utils.mk

.PHONY: build start test lint coverage

build:
	$(HIDE)docker build -f Dockerfile -t $(DOCKER_IMAGE) $(PWD)

start:
	$(HIDE)docker run --rm -it $(VOLUME) $(DOCKER_IMAGE) /bin/bash

test:
	$(HIDE) echo TODO: docker run test

lint:
	$(HIDE) echo TODO: docker run lint

coverage:
	$(HIDE) echo TODO: docker run coverage
