.PHONY: test 
.PHONY: push
.PHONY: push-and-test
.PHONY: clean

clean:
	sudo rm llama_weights/**/*.tensors
	sudo rm -rf gptq_weights
	sudo rm weights training_output.zip

test:
	./scripts/run_tests.sh

push:
	cog push r8.im/a16z-infra/llama-2-7b-chat

test-live:
	python test/push_test.py

push-and-test: push test-live
