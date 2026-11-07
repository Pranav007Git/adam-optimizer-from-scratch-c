# Optional: everything also works with plain python commands (see README).
PY ?= python3

all: validate run analyze

lib:
	gcc -O2 -shared -fPIC -o libnn.so src/nn.c -lm

validate:
	$(PY) validate.py

run:
	$(PY) run_experiments.py

analyze:
	$(PY) analyze.py

clean:
	rm -f libnn.so libnn.dylib libnn.dll
	rm -rf __pycache__
