all: reqs_optional/req_constraints.txt


#-- Build -------------------------------------------------------------
PACKAGE_VERSION  := `cat version.txt | tr -d '\n'`
CI_PYTHON_ENV    ?= $(shell dirname $(shell dirname `which python`))
PYTHON_PATH      ?= $(CI_PYTHON_ENV)/bin/python

.PHONY: reqs_optional/req_constraints.txt wheel venv publish

reqs_optional/req_constraints.txt:
	grep -v '#\|peft\|transformers\|accelerate' requirements.txt > $@

clean:
	rm -rf dist build h2ogpt.egg-info

wheel: clean
	docker run \
		--rm \
		--ulimit core=-1 \
		-v /home/0xdiag:/home/0xdiag:ro \
		-v /etc/passwd:/etc/passwd:ro \
		-v /etc/group:/etc/group:ro \
		-u `id -u`:`id -g` \
		-e HOME=/h2oai \
		-e HOST_HOSTNAME=`hostname` \
		-v `pwd`:/h2oai \
		--entrypoint bash \
		--workdir /h2oai \
		python:3.10 \
		-c "python3.10 setup.py bdist_wheel"

venv:
	rm -rf venv
	$(PYTHON_PATH) -m virtualenv -p $(PYTHON_PATH) venv

install-%:
	venv/bin/pip install dist/h2ogpt-$(PACKAGE_VERSION)-py3-none-any.whl[$*]

publish:
	echo "Publishing not implemented yet."

print-%:
	@echo $($*)

#-- Tests -------------------------------------------------------------
.PHONY: test

test:
	venv/bin/pytest tests --junit-xml=test_report.xml
