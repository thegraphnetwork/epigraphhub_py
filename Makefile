#* Variables
SHELL := /usr/bin/env bash

#* Docker variables
DOCKER=docker-compose --file docker/compose.yaml --env-file docker/.env
DOCKER_IMAGE:=epigraphhub_py
DOCKER_VERSION:=latest
DOCKER_SERVICES:=

#* Formatters
.PHONY: linter
linter:
	pre-commit run --all-files --verbose
	poetry run darglint --verbosity 2 epigraphhub_py tests

#* Tests

.PHONY: test
test:
	poetry run pytest -c pyproject.toml --cov-report=html --cov tests/
	poetry run coverage-badge -o assets/images/coverage.svg -f


.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive epigraphhub_py tests


#* Docker

# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(DOCKER_IMAGE):$(DOCKER_VERSION) ...
	docker build \
		-t $(DOCKER_IMAGE):$(DOCKER_VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(DOCKER_IMAGE):$(DOCKER_VERSION) ...
	docker rmi -f $(DOCKER_IMAGE):$(DOCKER_VERSION)

.PHONY: docker-compose-build
docker-compose-build:
	$(DOCKER) pull ${DOCKER_SERVICES}
	$(DOCKER) build ${DOCKER_SERVICES}

.PHONY: docker-compose-start
docker-compose-start:
	$(DOCKER) up -d ${DOCKER_SERVICES}

.PHONY: docker-compose-stop
docker-compose-stop:
	$(DOCKER) stop ${DOCKER_SERVICES}

.PHONY: docker-compose-restart
docker-compose-restart: docker-compose-stop docker-compose-stop

#* Cleaning

.PHONY: cleanup
cleanup:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E ".DS_Store" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	rm -rf build/
