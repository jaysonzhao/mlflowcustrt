# TODO: choose appropriate base image, install Python, MLServer, and
# dependencies of your MLModel implementation
FROM registry.redhat.io/ubi8/python-38:latest
RUN pip install mlserver
# ...

# The custom `MLModel` implementation should be on the Python search path
# instead of relying on the working directory of the image. If using a
# single-file module, this can be accomplished with:
COPY custom_model.py ./custom_model.py

# The environment variables here are for compatibility with ModelMesh Serving.
# These can also be set in the ServingRuntime, but this is recommended for
# consistency when building and testing
ENV MLSERVER_MODELS_DIR=/models/_mlserver_models \
    MLSERVER_GRPC_PORT=8001 \
    MLSERVER_HTTP_PORT=8002 \
    MLSERVER_LOAD_MODELS_AT_STARTUP=false \
    MLSERVER_MODEL_NAME=dummy-model

# With this setting, the implementation field is not required in the model
# settings which eases integration by allowing the built-in adapter to generate
# a basic model settings file
ENV MLSERVER_MODEL_IMPLEMENTATION=custom_model.CustomMLModel

CMD ["mlserver", "start", "${MLSERVER_MODELS_DIR}"]
