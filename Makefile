all: req_constraints.txt

.PHONY: req_constraints.txt
req_constraints.txt:
	grep -v '#\|peft' requirements.txt > req_constraints.txt
