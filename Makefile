all: clean

PACKAGE_VERSION       := `cat version.txt | tr -d '\n'`
PYTHON_BINARY         ?= `which python`
DOCKER_BINARY         ?= docker
DOCKER_BINARY_RUNTIME ?=
CMD_TO_RUN_IN_DOCKER  ?= make clean dist

BUILD_TAG_FILES      := requirements.txt Dockerfile `ls reqs_optional/*.txt | sort`
$(eval BUILD_TAG = $(shell md5sum $(BUILD_TAG_FILES) 2> /dev/null | sort | md5sum | cut -d' ' -f1))
DOCKER_TEST_IMAGE    := harbor.h2o.ai/h2ogpt/test-image:$(BUILD_TAG)

.PHONY: reqs_optional/req_constraints.txt publish dist test docker_build run_in_docker

reqs_optional/req_constraints.txt:
	grep -v '#\|peft\|transformers\|accelerate' requirements.txt > $@

clean:
	rm -rf dist build h2ogpt.egg-info

venv:
	$(PYTHON_BINARY) -m virtualenv -p $(PYTHON_BINARY) venv

install-%:
	$(PYTHON_BINARY) -m pip install dist/h2ogpt-$(PACKAGE_VERSION)-py3-none-any.whl[$*]

dist:
	$(PYTHON_BINARY) setup.py bdist_wheel

test:
	$(PYTHON_BINARY) -m pytest tests --junit-xml=test_report.xml

publish:
	echo "Publishing not implemented yet."

docker_build:
ifeq ($(shell curl --write-out %{http_code} -sS --output /dev/null -X GET http://harbor.h2o.ai/api/v2.0/projects/h2ogpt/repositories/test-image/artifacts/$(BUILD_TAG)/tags),200)
	@echo "Image already pushed to Harbor: $(DOCKER_TEST_IMAGE)"
else
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_TEST_IMAGE) -f Dockerfile .
	docker push $(DOCKER_TEST_IMAGE)
endif

run_in_docker:
	$(DOCKER_BINARY) run \
		$(DOCKER_BINARY_RUNTIME) \
		--rm \
		--security-opt seccomp=unconfined \
		--ulimit core=-1 \
		--entrypoint bash \
		--workdir /h2oai \
		-u `id -u`:`id -g` \
		-e HOME=/h2oai \
		-e PYTHON_BINARY=/usr/bin/python3.10 \
		-e USE_WHEEL=1 \
		-e PYTEST_TEST_NAME=$$PYTEST_TEST_NAME \
		-e IS_PR_BUILD=$$IS_PR_BUILD \
		-v /home/0xdiag:/home/0xdiag:ro \
		-v /etc/passwd:/etc/passwd:ro \
		-v /etc/group:/etc/group:ro \
		-v `pwd`:/h2oai \
		$(DOCKER_TEST_IMAGE) \
		-c "$(CMD_TO_RUN_IN_DOCKER)"

print-%:
	@echo $($*)
