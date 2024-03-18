all: clean dist

PACKAGE_VERSION       := `cat version.txt | tr -d '\n'`
BUILD_TAG             := $(shell git describe --always --dirty)
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

git_hash.txt:
	@echo "$(shell git rev-parse HEAD)" >> $@

# Deprecated for now, no 0.4.1 on pypi, use release binary wheel that has no CUDA errors anymore
docker_build_deps:
	@cp docker_build_script_ubuntu.sh docker_build_script_ubuntu.sh.back
	@sed -i '/# Install prebuilt dependencies/,$$d' docker_build_script_ubuntu.sh
	@docker build -t h2ogpt-deps-builder -f Dockerfile .
	@mv docker_build_script_ubuntu.sh.back docker_build_script_ubuntu.sh
	@mkdir -p prebuilt_deps
	@docker run \
		--rm \
		-it \
		--entrypoint bash \
		--runtime nvidia \
		-v `pwd`:/dot \
		-v /etc/passwd:/etc/passwd:ro \
		-v /etc/group:/etc/group:ro \
		-u `id -u`:`id -g` \
		h2ogpt-deps-builder  -c " \
			mkdir -p /dot/prebuilt_deps && cd /dot/prebuilt_deps && \
			GITHUB_ACTIONS=true python3.10 -m pip install auto-gptq==0.4.2 --no-cache-dir --use-deprecated=legacy-resolver && \
			python3.10 -m pip wheel auto-gptq==0.4.2 \
		"
	@docker run \
		--rm \
		-it \
		--entrypoint bash \
		-v `pwd`:/dot \
		quay.io/pypa/manylinux2014_x86_64 -c " \
			ln -s /usr/local/bin/python3.10 /usr/local/bin/python3 && cd /tmp && \
			git clone https://github.com/h2oai/duckdb.git && \
			cd duckdb && \
			git checkout dcd8c1ffc53dd020623630efb99ba6a3a4cbc5ad && \
			BUILD_PYTHON=1 make release && \
			cd tools/pythonpkg  && \
			python3.10 setup.py bdist_wheel  && \
			cp dist/duckdb-0.*.whl /dot/prebuilt_deps \
		"
	s3cmd put prebuilt_deps/auto_gptq-0.4.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl s3://artifacts.h2o.ai/deps/h2ogpt/ && \
	s3cmd setacl s3://artifacts.h2o.ai/deps/h2ogpt/auto_gptq-0.4.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --acl-public
	s3cmd put prebuilt_deps/duckdb-0.8.2.dev4026+gdcd8c1ffc5-cp310-cp310-linux_x86_64.whl s3://artifacts.h2o.ai/deps/h2ogpt/ && \
	s3cmd setacl s3://artifacts.h2o.ai/deps/h2ogpt/duckdb-0.8.2.dev4026+gdcd8c1ffc5-cp310-cp310-linux_x86_64.whl --acl-public

docker_build: build_info.txt
ifeq ($(shell curl --connect-timeout 4 --write-out %{http_code} -sS --output /dev/null -X GET http://harbor.h2o.ai/api/v2.0/projects/h2ogpt/repositories/test-image/artifacts/$(BUILD_TAG)/tags),200)
	@echo "Image already pushed to Harbor: $(DOCKER_TEST_IMAGE)"
else
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_TEST_IMAGE) -f Dockerfile .
	docker push $(DOCKER_TEST_IMAGE)
endif

just_docker_build: build_info.txt
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_TEST_IMAGE) -f Dockerfile .

docker_build_runner: docker_build
	-docker pull $(DOCKER_TEST_IMAGE)
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:$(BUILD_TAG)
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:$(PACKAGE_VERSION)
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:latest
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:$(BUILD_TAG)
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:$(PACKAGE_VERSION)
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:latest
ifdef BUILD_ID
	docker tag $(DOCKER_TEST_IMAGE) gcr.io/vorvan/h2oai/h2ogpt-runtime:$(PACKAGE_VERSION)-$(BUILD_ID)
	docker push gcr.io/vorvan/h2oai/h2ogpt-runtime:$(PACKAGE_VERSION)-$(BUILD_ID)
endif

print-%:
	@echo $($*)
