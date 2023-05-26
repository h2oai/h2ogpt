all: req_constraints.txt

.PHONY: req_constraints.txt
req_constraints.txt:
	grep -v '#\|peft\|transformers\|accelerate' requirements.txt > req_constraints.txt
