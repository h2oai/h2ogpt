all: clean dist

PACKAGE_VERSION       := `cat version.txt | tr -d '\n'`
BUILD_TAG_FILES       := requirements.txt Dockerfile `ls reqs_optional/*.txt | sort`
BUILD_TAG             := $(shell md5sum $(BUILD_TAG_FILES) 2> /dev/null | sort | md5sum | cut -d' ' -f1)
DOCKER_TEST_IMAGE     := harbor.h2o.ai/h2ogpt/test-image:$(BUILD_TAG)
DOCKER_RUN_IMAGE      := $(DOCKER_TEST_IMAGE)-runtime
PYTHON_BINARY         ?= `which python`
DEFAULT_MARKERS       ?= "not need_tokens and not need_gpu"

.PHONY: reqs_optional/req_constraints.txt venv dist test publish docker_build

reqs_optional/req_constraints.txt:
	grep -v '#\|peft\|transformers\|accelerate' requirements.txt > $@

clean:
	rm -rf dist build h2ogpt.egg-info

venv:
	$(PYTHON_BINARY) -m virtualenv -p $(PYTHON_BINARY) venv

install:
	$(PYTHON_BINARY) -m pip install dist/h2ogpt-$(PACKAGE_VERSION)-py3-none-any.whl

install-%:
	$(PYTHON_BINARY) -m pip install dist/h2ogpt-$(PACKAGE_VERSION)-py3-none-any.whl[$*]

dist:
	$(PYTHON_BINARY) setup.py bdist_wheel

test:
	$(PYTHON_BINARY) -m pip install requirements-parser -c reqs_optional/req_constraints.txt
	$(PYTHON_BINARY) -m pytest tests --disable-warnings --junit-xml=test_report.xml -m "$(DEFAULT_MARKERS)"

test_imports:
	$(PYTHON_BINARY) -m pytest tests/test_imports.py --disable-warnings --junit-xml=test_report.xml -m "$(DEFAULT_MARKERS)"

publish:
	echo "Publishing not implemented yet."

docker_build:
ifeq ($(shell curl --write-out %{http_code} -sS --output /dev/null -X GET http://harbor.h2o.ai/api/v2.0/projects/h2ogpt/repositories/test-image/artifacts/$(BUILD_TAG)/tags),200)
	@echo "Image already pushed to Harbor: $(DOCKER_TEST_IMAGE)"
else
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_TEST_IMAGE) -f Dockerfile .
	docker push $(DOCKER_TEST_IMAGE)
endif

.PHONY: Dockerfile-runner.dockerfile

Dockerfile-runner.dockerfile: Dockerfile-runner.in
	cat $< \
	| sed 's|BASE_DOCKER_IMAGE_SUBST|$(DOCKER_TEST_IMAGE)|g' \
	> $@

docker_build_runner: docker_build Dockerfile-runner.dockerfile
	docker pull $(DOCKER_TEST_IMAGE)
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_RUN_IMAGE) -f Dockerfile-runner.dockerfile .
	docker push $(DOCKER_RUN_IMAGE)
	docker tag $(DOCKER_RUN_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:$(BUILD_TAG)

print-%:
	@echo $($*)
