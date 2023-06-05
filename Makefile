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

print-%:
	@echo $($*)
