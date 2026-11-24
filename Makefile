.PHONY: build clean test test-c install dev

build:
	mkdir -p build
	cd build && cmake .. -DBUILD_TESTS=ON && cmake --build .

clean:
	rm -rf build dist *.egg-info

test-c: build
	cd build && ctest --output-on-failure

test:
	pytest tests/ -v

install:
	pip install .

dev:
	pip install -e ".[dev]"
