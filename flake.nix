{
  inputs.cognix.url = "github:datakami/cognix";

  outputs = { self, cognix, ... }:
    let
      packages = { insert-name-here = ./.; };
      callCognix = cognix.legacyPackages.x86_64-linux.callCognix;
      modelNames = builtins.attrNames (builtins.readDir ./models);
      # use the llama model at ./models/${name}
      llama-model = name:
        callCognix {
          inherit name;
          paths.projectRoot = ./.;
          # TODO: separate deps and lock.json for some models
          paths.lockFile = "lock.json";
          # Too many deps, make sure mlc ends up in its own layer
          # TODO: sort them based on size
          dockerTools.streamLayeredImage.maxLayers = 110;
          # ignore *.nix, add dockerignore contents
          cognix.sourceIgnores = builtins.concatStringsSep "\n" [
            "*.nix"
            (builtins.readFile ./model_templates/.dockerignore)
            (if builtins.pathExists ./models/${name}/.dockerignore then
              builtins.readFile ./models/${name}/.dockerignore
            else
              "")
          ];
          cognix.postCopyCommands = ''
            pushd $out/src
            model_dir=models/${config.name}
            find $model_dir -type f ! -path "$model_dir/model_artifacts/*" -exec ln -sf {} . \;
            [ -e $model_dir/.env ] && ln -sf $model_dir/.env .env || true
            popd
          '';
        } ./.;
      # evaluates all models in ./models
      evaluated_models = builtins.mapAttrs (name: _: llama-model name)
        (builtins.readDir ./models);
    in {
      packages.x86_64-linux = evaluated_models // {
        default = evaluated_models.llama-2-7b-mlc;
      };
      devShells.x86_64-linux.default = cognix.devShells.x86_64-linux.default;
      apps.x86_64-linux.default = {
        type = "app";
        program = "${cognix.packages.x86_64-linux.default}/bin/cognix";
      };
    };
}
