.PHONY: test 
.PHONY: push
.PHONY: push-and-test
.PHONY: clean

clean:
	sudo rm llama_weights/**/*.tensors
	sudo rm -rf default_base_weights
	sudo rm weights training_output.zip

test:
	./test/run_tests.sh

push:
	cog push r8.im/a16z-infra/llama-2-7b-chat

test-live:
	python3 test/push_test.py

push-and-test: push test-live
