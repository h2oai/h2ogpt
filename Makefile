all: clean dist

PACKAGE_VERSION       := `cat version.txt | tr -d '\n'`
BUILD_TAG_FILES       := requirements.txt Dockerfile `ls reqs_optional/*.txt | sort`
BUILD_TAG             := $(shell md5sum $(BUILD_TAG_FILES) 2> /dev/null | sort | md5sum | cut -d' ' -f1)
DOCKER_TEST_IMAGE     := harbor.h2o.ai/h2ogpt/test-image:$(BUILD_TAG)
PYTHON_BINARY         ?= `which python`

.PHONY: reqs_optional/req_constraints.txt venv dist test publish docker_build

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

has_nvidia_smi    := $(shell nvidia-smi >/dev/null 2>&1 && echo "WORKING")
ifeq ($(has_nvidia_smi),)
	override DOCKER_BINARY=docker
	override DOCKER_BINARY_RUNTIME=
    $(warning System has no GPUs, using DOCKER_BINARY=$(DOCKER_BINARY) and DOCKER_BINARY_RUNTIME=$(DOCKER_BINARY_RUNTIME))
else
	override DOCKER_BINARY?=nvidia-docker
    ifeq ($(shell echo `which $(DOCKER_BINARY)`),)
        override DOCKER_BINARY=docker
    endif
	override DOCKER_BINARY_RUNTIME=--runtime nvidia
    $(warning System has GPUs, using DOCKER_BINARY=$(DOCKER_BINARY) and DOCKER_BINARY_RUNTIME=$(DOCKER_BINARY_RUNTIME))
endif
test_in_docker: docker_build
	$(DOCKER_BINARY) run \
		$(DOCKER_BINARY_RUNTIME) \
		--rm \
		--init \
		--workdir /h2oai \
		--entrypoint bash \
		-u `id -u`:`id -g` \
		-e HOME=/h2oai \
		-e HOST_HOSTNAME=`hostname` \
		-v /etc/passwd:/etc/passwd:ro \
		-v /etc/group:/etc/group:ro \
		-v `pwd`:/h2oai \
		$(DOCKER_TEST_IMAGE) \
		-c "nvidia-smi || true && \
			python3.10 -m pip install -r requirements.txt && \
		   	python3.10 -m pip install -r reqs_optional/requirements_optional_4bit.txt && \
		   	python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt && \
		   	python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt && \
		   	python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt && \
		   	python3.10 -m pytest tests --junit-xml=test_report.xml"

print-%:
	@echo $($*)
