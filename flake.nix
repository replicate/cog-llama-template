{
  inputs = {
    cognix.url = "github:technillogue/cognix?rev=4017831c8cdbb8bc3d32eec0887bcae5e5815639";
  };

  outputs = { self, cognix }@inputs: cognix.lib.singleCognixFlake inputs "insert-name-here";
}
