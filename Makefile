# make build
# make start

SHELL := /bin/bash
HIDE ?= @
DOCKER_IMAGE ?= brain/bert
DOCKER_CONTAINER ?= bert
VOLUME ?=-v $(PWD):/brain/src -v $(DOCKER_CONTAINER)-venv:/venv
ENV ?= --env-file=./docker/dev-env.rc

-include ./docker/registry.mk
-include ./docker/utils.mk
-include ./docker/docs.mk

.PHONY: build install start test lint coverage


build:
	$(HIDE)docker build --build-arg GITHUB_TOKEN=$(GITHUB_TOKEN) -f Dockerfile -t $(DOCKER_IMAGE) $(PWD)
	$(HIDE)$(MAKE) install

install:
	$(HIDE)docker run --rm \
		$(VOLUME) \
		-e GITHUB_TOKEN=$(GITHUB_TOKEN) \
		-e ENVIRONMENT=development $(DOCKER_IMAGE) ./docker/setup.sh

start: .deps
	$(HIDE)docker-compose -f docker/docker-compose.yml up --no-deps $(DOCKER_CONTAINER)

# Database Targets
mysql: .deps
	$(HIDE)docker-compose -f docker/docker-compose.yml up -d create-db

migrate.%: .deps
	$(HIDE)ENVIRONMENT=$* docker-compose -f docker/docker-compose.yml up migrations

migrate:
	-$(HIDE)$(MAKE) migrate.development

test: .deps
	$(HIDE)ENVIRONMENT=test docker-compose -f ./docker/docker-compose.yml up test
	$(HIDE)exit `docker wait $(DOCKER_CONTAINER)-test`
	$(HIDE)docker-compose -f ./docker/docker-compose.yml rm -fsv

lint:
	$(HIDE)docker run -it --rm $(VOLUME) $(DOCKER_IMAGE) flake8 .

coverage: .deps
	$(HIDE)ENVIRONMENT=test docker-compose -f ./docker/docker-compose.yml up -d coverage
	$(HIDE)docker-compose -f ./docker/docker-compose.yml logs coverage
	$(HIDE)exit `docker wait $(DOCKER_CONTAINER)-coverage`
	$(HIDE)docker-compose -f ./docker/docker-compose.yml rm -fsv
