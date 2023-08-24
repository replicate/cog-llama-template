# `cog-llama-template` Model Management

The `cog-llama-template` repo decomposes model management into four constructs:

* **Templates.** We store templates in the `./model_templates/` directory. For our purposes, a template includes the following model specific artifacts: `cog.yaml`, `config.py`, `predict.py`. 

* **Models.** We store artifacts for initialized models in the `./models/` directory. These artifacts are copied from a template and then updated with model specific information. 

* **Shared code.** Models defined in `cog-llama-template` share code, e.g. implementations of training and inference methods. Shared code is maintained in the `./src/` directory. 

* **Active model.** To build, run, or push a specific model, it's artifacts must be copied from its associated `./models/` directory to the root of this project. We do this so that `./src/` code is available at build time. We refer to this copying process as model *selection*.

To help users manage and interact with these constructs, we provide a `Makefile` with commands that streamline the model development process. Below, is a step-by-step demonstration of how you can use the `Makefile` to develop a model. 

**1. Initialize a new model.**

You can initialize a new model by setting the environment variable `SELECTED_MODEL` to the name of the model you want to initialize. The name is arbitrary and there are no forced naming conventions, however our inhouse style is lowered dash-case. 

The `SELECTED_MODEL` environment variable will be referenced for all subsequent make commands. However, you can also specify the argument `name=<model_name>` instead of setting an environment variable. 

Finally, `make init` will copy a model template from `model_templates` to `./models/<model-name>/`.

```
export SELECTED_MODEL=llama-2-70b-chat
make init
```

**2. Update model details.**

Currently, you need to manually update model details in `config.py`, as well as possibly in `predict.py`. Specifically, you need to provide variables for global config variables that determine inference logic and file's that should be downloaded. 

We assume that model artifacts are stored in an accessible and external location. During `setup` or training intialization, model artifacts specified in `config.py` will be downloaded. 

However, in some cases, it is preferable to not expose the locations of model artifacts in `config.py`. In such cases, you can store information in a `.env` file in your model's directory. At runtime, those environment variables will be initialized and their values will be used by `config.py`. 

For example, we store paths to model artifacts in `.env` and load this at runtime.

**3. Select model.**

To interact with a model, its artifacts need to be copied to root of `cog-llama-templates`. You can do this like:

 ```make select``` 

 or 

 ```make select model=<model-name>```

 This will copy the model artifacts to root and run `cog build`. 

**Local testing.**

Our `Makefile` provides easy access to a rudimentary test suite that supports local and staged testing.

Assuming you've set the `SELECTED_MODEL` environment variable, you can just call:

`make test-local`

Appending `verbose=true` will run tests with `-s` so that output will be printed.

**Staging.**

We also provide a staging workflow via `make stage` and `make test-stage-<...>`. To use the staging commands, you must specify your Replicate user account (we default to `replicate-internal`) and create a Replicate model in the specified account with the naming convention `staging-<$SELECTED_MODEL>`. Accordingly, if your selected model is `llama-2-7b`, you would create a model called `staging-llama-2-7b`. 

You also need to log in via cog login and set the `REPLICATE_API_TOKEN` environment variable to your accounts API token. 

Calling `make stage` will push the selected model to the associated staging model. Then you can call `make test-stage`. 





