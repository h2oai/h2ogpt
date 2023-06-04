all: reqs_optional/req_constraints.txt

PACKAGE_VERSION      := `cat version.txt | tr -d '\n'`
PYTHON_BINARY        ?= `which python`
CMD_TO_RUN_IN_DOCKER ?= "make clean dist"

BUILD_TAG_FILES   := requirements.txt Dockerfile `ls reqs_optional/*.txt | sort`

$(eval BUILD_TAG = $(shell md5sum $(BUILD_TAG_FILES) 2> /dev/null | sort | md5sum  | cut -d' ' -f1))
DOCKER_TEST_IMAGE := harbor.h2o.ai/h2ogpt/test-image:$(BUILD_TAG)

ifndef SKIP_BUILD
include ci/build.mk
endif

.PHONY: reqs_optional/req_constraints.txt publish dist test

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

run_in_docker: export SKIP_BUILD=1
run_in_docker:
	docker run \
		$(DOCKER_BINARY_RUNTIME) \
		--rm \
		--ulimit core=-1 \
		-v /home/0xdiag:/home/0xdiag:ro \
		-v /etc/passwd:/etc/passwd:ro \
		-v /etc/group:/etc/group:ro \
		-u `id -u`:`id -g` \
		-e "HOME=/h2oai" \
		-e "PYTHON_BINARY=/usr/bin/python3.10" \
		-e "BUILD_NUMBER=$$BUILD_NUMBER" \
		-v `pwd`:/h2oai \
		--entrypoint bash \
		--workdir /h2oai \
		$(DOCKER_TEST_IMAGE) \
		-c "$(CMD_TO_RUN_IN_DOCKER)"

print-%:
	@echo $($*)
