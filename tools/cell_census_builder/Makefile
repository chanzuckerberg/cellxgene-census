# Build docker image. This Makefile is for convenience in development,
# and as a means to manually build in advance of pushing the image to
# a registry.
#
# COMING SOON: Docker builds for routine use are created by a GHA, and
# will be available in a Docker repository.

# Create the image
.PHONY: image
image: clean
	python3 -m build .
	docker build --build-arg=COMMIT_SHA=$(git rev-parse --short HEAD) -t cell-census-builder .

# Clean Python build
.PHONY: clean
clean:
	rm -rf build dist

# Prune docker cache
.PHONY: prune
prune:
	docker system prune -f
	if [ "$(docker ps -aq)" ]; then docker rm -f $(docker ps -aq) ; fi
	if [ "$(docker images -q)" ]; then docker rmi -f $(docker images -q) ; fi
