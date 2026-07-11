# =============================================================================
#  Almo – Makefile for common development tasks
# =============================================================================
#
#  Usage: make <target> [CONFIG=<config_name>]
#
#  Examples:
#    make setup              # full setup (build, start, data import, dbt, verify)
#    make train              # train with default config (REG)
#    make train CONFIG=CLASS # train classifier
#    make simulator-up       # start the traffic simulator (requires trained models)
#    make simulator-down     # stop the traffic simulator
#    make demo               # run the COVID drift demo
#    make help               # show all commands
# =============================================================================

COMPOSE_FILE := docker/compose.yml
COMPOSE_CMD  := docker compose -f $(COMPOSE_FILE)

# Default configuration names
DEFAULT_TRAIN_CONFIG := REG
DEFAULT_TUNE_CONFIG  := OPTUNA_REG

# If CONFIG is not set, use the default
CONFIG ?= $(DEFAULT_TRAIN_CONFIG)

# Default target: show help
.DEFAULT_GOAL := help

# --------------------------------------------------------------------------
#  Help
# --------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Optional variables:"
	@echo "  CONFIG=<name>  - configuration name (from flows/config.py)"
	@echo "                   default for train: $(DEFAULT_TRAIN_CONFIG)"
	@echo "                   default for tune : $(DEFAULT_TUNE_CONFIG)"
	@echo ""
	@echo "Examples:"
	@echo "  make train CONFIG=MY_NEW_MODEL"
	@echo "  make tune CONFIG=OPTUNA_CLASS"

# --------------------------------------------------------------------------
#  Setup (full system bootstrap)
# --------------------------------------------------------------------------
.PHONY: setup
setup:  ## Run the complete setup script (build, start, import, dbt, verify)
	./setup.sh

# --------------------------------------------------------------------------
#  Service Management
# --------------------------------------------------------------------------
.PHONY: up
up:  ## Start all services (detached)
	$(COMPOSE_CMD) up -d

.PHONY: down
down:  ## Stop all services and remove containers
	$(COMPOSE_CMD) down

.PHONY: logs
logs:  ## Tail logs from all services
	$(COMPOSE_CMD) logs -f

.PHONY: ps
ps:  ## Show service status
	$(COMPOSE_CMD) ps

# --------------------------------------------------------------------------
#  Simulator
# --------------------------------------------------------------------------
.PHONY: simulator-up
simulator-up:  ## Start the traffic simulator (requires trained models)
	$(COMPOSE_CMD) --profile manual up -d simulator

.PHONY: simulator-down
simulator-down:  ## Stop the traffic simulator
	$(COMPOSE_CMD) stop simulator > /dev/null 2>&1
	$(COMPOSE_CMD) rm -f simulator > /dev/null 2>&1

.PHONY: simulator-logs
simulator-logs:  ## Tail logs from the simulator
	$(COMPOSE_CMD) logs -f simulator

# --------------------------------------------------------------------------
#  Training
# --------------------------------------------------------------------------
.PHONY: train
train:  ## Train a model with the specified CONFIG (default: REG)
	$(COMPOSE_CMD) exec -e PYTHONPATH=/app api python flows/train_flow.py $(CONFIG)

.PHONY: train-reg
train-reg:  ## Train regression model (config: REG)
	$(MAKE) train CONFIG=REG

.PHONY: train-class
train-class:  ## Train classification model (config: CLASS)
	$(MAKE) train CONFIG=CLASS

# --------------------------------------------------------------------------
#  Tuning (Optuna)
# --------------------------------------------------------------------------
.PHONY: tune
tune:  ## Run Optuna tuning with the specified CONFIG (default: OPTUNA_REG)
	$(COMPOSE_CMD) exec -e PYTHONPATH=/app api python flows/tune_flow.py $(CONFIG)

.PHONY: tune-reg
tune-reg:  ## Run Optuna tuning for regression (config: OPTUNA_REG)
	$(MAKE) tune CONFIG=OPTUNA_REG

.PHONY: tune-class
tune-class:  ## Run Optuna tuning for classification (config: OPTUNA_CLASS)
	$(MAKE) tune CONFIG=OPTUNA_CLASS

# --------------------------------------------------------------------------
#  Demo
# --------------------------------------------------------------------------
.PHONY: demo
demo:  ## Run the COVID data drift demo
	./demo/covid_data_drift_demo.sh

# --------------------------------------------------------------------------
#  Cleanup
# --------------------------------------------------------------------------
.PHONY: clean
clean:  ## Stop all containers and remove volumes (deletes all data!)
	$(COMPOSE_CMD) down -v

.PHONY: reset
reset: clean  ## Alias for clean