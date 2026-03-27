VERSION ?= $(shell git describe --tags --abbrev=0 2>/dev/null || echo v2)

build:
	docker buildx build --platform linux/amd64 -t particle-core:$(VERSION) -f Dockerfile .
	mkdir -p ../particle-run/images
	docker save particle-core:$(VERSION) -o ../particle-run/images/particle-core_$(VERSION).tar
