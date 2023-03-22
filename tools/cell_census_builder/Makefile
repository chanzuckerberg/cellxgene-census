# Build docker container

.PHONY: container
container:
	python3 -m build .
	docker build --build-arg=COMMIT_SHA=$(git rev-parse --short HEAD) -t cell-census-builder .

.PHONY: clean
clean:
	rm -rf build dist
