#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ki-go
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Make helper 

random-bvb:
	$(PYTHON_INTERPRETER) src/random_bvb.py

random-hvb:
	$(PYTHON_INTERPRETER) src/random_hvb.py

random-web:
	$(PYTHON_INTERPRETER) src/random_web.py

depthbruned-bvb:
	$(PYTHON_INTERPRETER) src/depthbruned_bvb.py

alphabeta-bvb:
	$(PYTHON_INTERPRETER) src/alphabeta_bvb.py

mcts-bvb:
	$(PYTHON_INTERPRETER) src/mcts_bvb.py

mcts-hvb:
	$(PYTHON_INTERPRETER) src/mcts_hvb.py

mcts-web:
	$(PYTHON_INTERPRETER) src/mcts_web.py

get-sgf:
	$(PYTHON_INTERPRETER) src/get_sgf.py

train-betago:
	$(PYTHON_INTERPRETER) src/train_betago.py

test-betago:
	$(PYTHON_INTERPRETER) src/test_betago.py

betago-hvb:
	$(PYTHON_INTERPRETER) src/betago_hvb.py

betago-web:
	$(PYTHON_INTERPRETER) src/betago_web.py

betago-local:
	$(PYTHON_INTERPRETER) src/betago_local.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Test python environment is setup correctly
test-environment:
	$(PYTHON_INTERPRETER) test_environment.py


.PHONY: clean data features lint 


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
#
help:
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

.PHONY: help
