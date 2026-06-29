.PHONY: install dev test image convert validate clean

install:        ## create venv + install
	./install.sh

dev:            ## install with dev deps + run tests
	./install.sh --dev

test:           ## run the test suite (assumes deps installed)
	pytest -q

image:          ## generate the synthetic test portrait
	photo2cricut-makeimg examples/test_portrait.jpg

convert: image  ## run the demo conversion
	photo2cricut examples/test_portrait.jpg examples/test_portrait.svg --method xdog --width-in 8

validate:       ## validate the demo output
	photo2cricut-validate examples/test_portrait.svg

clean:          ## remove venv, caches, and generated examples
	rm -rf .venv build dist *.egg-info .pytest_cache
	rm -f examples/*.jpg examples/*.png examples/*.svg
