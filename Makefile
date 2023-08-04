all: clean dist

PACKAGE_VERSION       := `cat version.txt | tr -d '\n'`
BUILD_TAG_FILES       := requirements.txt Dockerfile `ls reqs_optional/*.txt | sort`
BUILD_TAG             := $(shell md5sum $(BUILD_TAG_FILES) 2> /dev/null | sort | md5sum | cut -d' ' -f1)
DOCKER_TEST_IMAGE     := harbor.h2o.ai/h2ogpt/test-image:$(BUILD_TAG)
PYTHON_BINARY         ?= `which python`
DEFAULT_MARKERS       ?= "not need_tokens and not need_gpu"

.PHONY: venv dist test publish docker_build build_info.txt

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

docker_build_deps:
	@rm -rf Dockerfile_deps
	@sed '/# Install prebuilt dependencies/,$$d' Dockerfile > Dockerfile_deps
	@docker build -t h2ogpt-deps-builder -f Dockerfile_deps .
	@docker run \
		--rm -it --entrypoint bash --runtime nvidia -v `pwd`:/dot \
		h2ogpt-deps-builder -c " \
			mkdir -p /dot/prebuilt_deps && cd /dot/prebuilt_deps && \
			GITHUB_ACTIONS=true python3.10 -m pip install auto-gptq==0.3.0 --no-cache-dir --use-deprecated=legacy-resolver && \
			python3.10 -m pip wheel auto-gptq==0.3.0 && \
		"
	s3cmd put prebuilt_deps/auto_gptq-0.3.0-cp310-cp310-linux_x86_64.whl s3://artifacts.h2o.ai/deps/h2ogpt/ && \
	s3cmd setacl s3://artifacts.h2o.ai/deps/h2ogpt/auto_gptq-0.3.0-cp310-cp310-linux_x86_64.whl --acl-public

docker_build: build_info.txt
ifeq ($(shell curl --connect-timeout 4 --write-out %{http_code} -sS --output /dev/null -X GET http://harbor.h2o.ai/api/v2.0/projects/h2ogpt/repositories/test-image/artifacts/$(BUILD_TAG)/tags),200)
	@echo "Image already pushed to Harbor: $(DOCKER_TEST_IMAGE)"
else
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_TEST_IMAGE) -f Dockerfile .
	docker push $(DOCKER_TEST_IMAGE)
endif

docker_build_runner: docker_build
	-docker pull $(DOCKER_TEST_IMAGE)
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:$(BUILD_TAG)
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:$(PACKAGE_VERSION)
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:latest
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:$(BUILD_TAG)
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:$(PACKAGE_VERSION)
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:latest

print-%:
	@echo $($*)
