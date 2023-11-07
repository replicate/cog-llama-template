{ config, ...}:
let
  pipDrvs = config.python-env.pip.drvs;
  sitePackages = drv: "${drv.public}/${config.python-env.deps.python.sitePackages}";
in
{
  cog.build.cog_version = "0.9.0-beta9";
  python-env.pip.drvs = {
    # remove unlinkable libs to fix autoPatchelf complaint
    torch = {
      mkDerivation.postInstall = ''
        rm $out/lib/python*/site-packages/torch/lib/lib{caffe2_nvrtc,torch_cuda_linalg}.so
      '';
    };
    # add dep on cudart bundled in torch
    mlc-ai-nightly-cu118 = { config, ... }: {
      env.autoPatchelfIgnoreMissingDeps = ["libcuda.so.1"];
      deps.torch = pipDrvs.torch;
      mkDerivation.postInstall = ''
        addAutoPatchelfSearchPath ${sitePackages config.deps.torch}/torch/lib
      '';
    };
    # add dep on tvm bundled by mlc-ai
    # TODO: xformers can find transient lib deps, how does that work?
    mlc-chat-nightly-cu118 = { config, ... }: {
      deps.mlc-ai = pipDrvs.mlc-ai-nightly-cu118;
      mkDerivation.postInstall = ''
        addAutoPatchelfSearchPath ${sitePackages config.deps.mlc-ai}/tvm
      '';
    };
    # TODO: xformers depends on cuda 12
    # xformers.env.autoPatchelfIgnoreMissingDeps = [ "libcudart.so.12" ];
  };
}

