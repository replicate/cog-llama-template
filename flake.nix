{
  inputs = {
    cognix.url = "github:datakami/cognix";
  };

  outputs = { self, cognix }@inputs: cognix.lib.singleCognixFlake inputs "insert-name-here";
}
