from typing import List
from mlserver import MLModel, types
from mlserver.utils import get_model_uri

class CustomMLModel(MLModel):
  async def load(self) -> bool:
    #model_uri = await get_model_uri(self._settings)
    #self._load_model_from_file(model_uri)
    #self.ready = True
    print("in loading overrided")
    return True

  async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:

    payload = self._check_request(payload)

    return types.InferenceResponse(
      model_name=self.name,
      model_version=self.version,
      outputs=self._predict_outputs(payload),
    )

    def _load_model_from_file(self, file_uri):
        # TODO: load model from file and instantiate class data
        print("loading: "+file_uri)
        return

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        # TODO: validate request: number of inputs, input tensor names/types, etc.
        return payload

    def _predict_outputs(self, payload: types.InferenceRequest) -> List[types.ResponseOutput]:
        inputs = payload.inputs
        # TODO: transform inputs into internal data structures
        # TODO: send data through the model's prediction logic
        outputs = []
        return outputs
