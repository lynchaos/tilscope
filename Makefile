.PHONY: install dev report test clean
install:        ## install the package
	pip install -e .
dev:            ## install with all extras + dev tools
	pip install -e ".[sc,llm,dev]"
report:         ## generate the demo report
	tilscope run --out examples/report.html --seed 0
test:           ## run the test suite
	pytest -q
clean:
	rm -rf build dist *.egg-info src/*.egg-info **/__pycache__
