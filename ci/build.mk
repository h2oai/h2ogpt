.PHONY: docker_build

ifeq ($(shell nvidia-smi > /dev/null 2>&1 ; echo $$?),0)
	override DOCKER_BINARY?=nvidia-docker
    ifeq ($(shell echo `which $(DOCKER_BINARY)`),)
        override DOCKER_BINARY=docker
    endif
	override DOCKER_BINARY_RUNTIME=--runtime nvidia
else
	override DOCKER_BINARY=docker
	override DOCKER_BINARY_RUNTIME=
endif

docker_build:
ifeq ($(shell docker pull $(DOCKER_TEST_IMAGE) > /dev/null 2>&1 ; echo $$?),0)
	-echo "pulled image $(DOCKER_TEST_IMAGE)"
else
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_TEST_IMAGE) -f Dockerfile .
	docker push $(DOCKER_TEST_IMAGE)
endif
