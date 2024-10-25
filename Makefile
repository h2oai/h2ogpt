all: clean dist

PACKAGE_VERSION              := `cat version.txt | tr -d '\n'`
BUILD_TAG                    := $(shell git describe --always --dirty)
DOCKER_H2OGPT_RUNTIME_IMAGE  := gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:$(BUILD_TAG)
DOCKER_H2OGPT_VLLM_IMAGE     := gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:$(BUILD_TAG)
PYTHON_BINARY                ?= `which python`
DEFAULT_MARKERS              ?= "not need_tokens and not need_gpu"

# h2ogpt base and vllm images built elsewhere and referenced here:
DOCKER_BASE_OS_IMAGE     := gcr.io/vorvan/h2oai/h2ogpt-oss-wolfi-base:9
DOCKER_VLLM_IMAGE        := gcr.io/vorvan/h2oai/h2ogpte-vllm:0.6.3.post1-38ed4ff2


.PHONY: venv dist test publish docker_build docker_push build_info.txt

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
	$(PYTHON_BINARY) -m pip install requirements-parser
	$(PYTHON_BINARY) -m pytest tests --disable-warnings --junit-xml=test_report.xml -m "$(DEFAULT_MARKERS)"

test_imports:
	$(PYTHON_BINARY) -m pytest tests/test_imports.py --disable-warnings --junit-xml=test_report.xml -m "$(DEFAULT_MARKERS)"

publish:
	echo "Publishing not implemented yet."

build_info.txt:
	@rm -rf build_info.txt
	@echo "commit=\"$(shell git rev-parse HEAD)\"" >> $@
	@echo "branch=\"`git rev-parse HEAD | git branch -a --contains | grep -v detached | sed -e 's~remotes/origin/~~g' -e 's~^ *~~' | sort | uniq | tr '*\n' ' '`\"" >> $@
	@echo "describe=\"`git describe --always --dirty`\"" >> $@
	@echo "build_os=\"`uname -a`\"" >> $@
	@echo "build_machine=\"`hostname`\"" >> $@
	@echo "build_date=\"$(shell date "+%Y%m%d")\"" >> $@
	@echo "build_user=\"`id -u -n`\"" >> $@
	@echo "base_version=\"$(PACKAGE_VERSION)\"" >> $@


docker_build: build_info.txt
ifeq ($(shell curl --connect-timeout 4 --write-out %{http_code} -sS --output /dev/null -X GET https://gcr.io/v2/vorvan/h2oai/h2oai-h2ogpt-runtime/manifests/$(BUILD_TAG)),200)
	@echo "Image already pushed to GCR: $(DOCKER_H2OGPT_RUNTIME_IMAGE)"
	docker pull $(DOCKER_H2OGPT_RUNTIME_IMAGE)
else
	docker pull $(DOCKER_BASE_OS_IMAGE)
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_H2OGPT_RUNTIME_IMAGE) -t h2ogpt:current -f Dockerfile .
endif
ifeq ($(shell curl --connect-timeout 4 --write-out %{http_code} -sS --output /dev/null -X GET https://gcr.io/v2/vorvan/h2oai/h2oai-h2ogpt-vllm/manifests/$(BUILD_TAG)),200)
	@echo "VLLM Image already pushed to GCR: $(DOCKER_H2OGPT_VLLM_IMAGE)"
	docker pull $(DOCKER_H2OGPT_VLLM_IMAGE)
else
	docker pull $(DOCKER_VLLM_IMAGE)
	docker tag $(DOCKER_VLLM_IMAGE) $(DOCKER_H2OGPT_VLLM_IMAGE)
endif

docker_push:
	docker tag $(DOCKER_H2OGPT_RUNTIME_IMAGE) gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:$(PACKAGE_VERSION)
	docker tag $(DOCKER_H2OGPT_VLLM_IMAGE) gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:$(PACKAGE_VERSION)

	docker tag $(DOCKER_H2OGPT_RUNTIME_IMAGE) gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:latest
	docker tag $(DOCKER_H2OGPT_VLLM_IMAGE) gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:latest

	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:$(BUILD_TAG)
	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:$(PACKAGE_VERSION)
	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:latest

	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:$(BUILD_TAG)
	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:$(PACKAGE_VERSION)
	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:latest

ifdef BUILD_ID
	docker tag $(DOCKER_H2OGPT_RUNTIME_IMAGE) gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:$(PACKAGE_VERSION)-$(BUILD_ID)
	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-runtime:$(PACKAGE_VERSION)-$(BUILD_ID)

	docker tag $(DOCKER_H2OGPT_VLLM_IMAGE) gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:$(PACKAGE_VERSION)-$(BUILD_ID)
	docker push gcr.io/vorvan/h2oai/h2oai-h2ogpt-vllm:$(PACKAGE_VERSION)-$(BUILD_ID)
endif

print-%:
	@echo $($*)
