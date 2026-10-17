PY ?= python3

# Keep freed NumPy buffers in the heap (glibc only; ignored on macOS). See Dockerfile.
export MALLOC_MMAP_THRESHOLD_ = 134217728
export MALLOC_TRIM_THRESHOLD_ = 536870912
export MALLOC_TOP_PAD_ = 268435456

.PHONY: setup tables train benchmark test run docker-build docker-run deploy teardown

setup:            ## install everything for training, tests and the service
	$(PY) -m pip install -r requirements-dev.txt

tables:           ## rebuild CIE tables and ColorChecker spectra (needs colour-science)
	$(PY) scripts/build_tables.py

train:            ## train the models, about 3 minutes on an M2
	$(PY) -W ignore scripts/train.py

benchmark:        ## AI search vs every set, lab trials, legacy formulas; about 10 minutes
	$(PY) -W ignore scripts/benchmark.py

test:
	$(PY) -m pytest -q

run:              ## http://localhost:8080
	$(PY) -m uvicorn service.main:app --reload --port 8080

docker-build:
	docker build -t firstshot .

docker-run:
	docker run --rm -p 8080:8080 firstshot

deploy:           ## AWS Lambda + Function URL, ap-south-1 by default
	./deploy/deploy.sh

teardown:
	./deploy/teardown.sh
