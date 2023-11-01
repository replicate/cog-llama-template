# Contributing

Thanks for taking the time to contribute to this project!

## Releases

This section documents the process used internally at Replicate to deploy the many variant Llama models.

Model variants live in the [models](models) directory, and deployment is managed by a [Makefile](Makefile).

To release a new model:

1. Run `make select <model-name>`, where model name corresponds to the name of a folder in the [models](models) directory, like `model-llama-2-7b`. This will copy stuff around and jigger the local state of the repo to say "use this model".
1. Run `make test-local` to test locally (assuming you're on a machine with a GPU).
1. Run `make stage test-stage <model-name>` to push to staging. If this passes, the model is ready to be promoted to production.
1. Run `REPLICATE_USER=replicate && make push test-prod <model-name>`. This runs the same tests as staging.

After releasing to production:

1. Search for old instances of the previous version's Docker image id in documentation and replace them with the new version.