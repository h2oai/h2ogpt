all: reqs_optional/req_constraints.txt

.PHONY: reqs_optional/req_constraints.txt
reqs_optional/req_constraints.txt:
	grep -v '#\|peft\|transformers\|accelerate' requirements.txt > $@
