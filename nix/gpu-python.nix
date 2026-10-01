{ pkgs
, lib
, uv2nix
, pyproject-nix
, pyproject-build-systems
, workspaceRoot
}:
# Lockfile-backed Python closure for the GPU SSV image. Each locked wheel is
# its own Nix store path so streamLayeredImage layers packages independently.
# The editable project is rebuilt from pyproject.toml, README, and src/.
let
  python = pkgs.python312;
  workspace = uv2nix.lib.workspace.loadWorkspace { inherit workspaceRoot; };
  overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
  projectName = "ip-claim";
  projectSrc = lib.fileset.toSource {
    root = workspaceRoot;
    fileset = lib.fileset.unions [
      (workspaceRoot + "/pyproject.toml")
      (workspaceRoot + "/README.md")
      (workspaceRoot + "/src")
    ];
  };
  pyprojectOverrides = final: prev:
    let
      inherit (final) resolveBuildSystem;
      withBuildSystems = lib.mapAttrs
        (name: spec: prev.${name}.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ resolveBuildSystem spec;
        }))
        {
          antlr4-python3-runtime = { setuptools = [ ]; };
        };
      cudaDriverMissing = [
        "libcuda.so.1"
        "libnvidia-ml.so.1"
        "libmlx5.so.1"
        "librdmacm.so.1"
        "libibverbs.so.1"
        "libpmix.so.2"
        "liboshmem.so.40"
        "libucs.so.0"
        "libucp.so.0"
        "libmpi.so.40"
        "libfabric.so.1"
      ];
      cudaWheelLibs = [
        { name = "nvidia-cublas-cu12"; sub = "cublas"; }
        { name = "nvidia-cuda-cupti-cu12"; sub = "cuda_cupti"; }
        { name = "nvidia-cuda-nvrtc-cu12"; sub = "cuda_nvrtc"; }
        { name = "nvidia-cuda-runtime-cu12"; sub = "cuda_runtime"; }
        { name = "nvidia-cudnn-cu12"; sub = "cudnn"; }
        { name = "nvidia-cufft-cu12"; sub = "cufft"; }
        { name = "nvidia-cufile-cu12"; sub = "cufile"; }
        { name = "nvidia-curand-cu12"; sub = "curand"; }
        { name = "nvidia-cusolver-cu12"; sub = "cusolver"; }
        { name = "nvidia-cusparse-cu12"; sub = "cusparse"; }
        { name = "nvidia-cusparselt-cu12"; sub = "cusparselt"; }
        { name = "nvidia-nccl-cu12"; sub = "nccl"; }
        { name = "nvidia-nvjitlink-cu12"; sub = "nvjitlink"; }
        { name = "nvidia-nvshmem-cu12"; sub = "nvshmem"; }
        { name = "nvidia-nvtx-cu12"; sub = "nvtx"; }
      ];
      addNvidiaLibSearch = deps: lib.concatMapStrings
        (dep: ''
          addAutoPatchelfSearchPath ${final.${dep.name}}/${python.sitePackages}/nvidia/${dep.sub}/lib
        '')
        deps;
      nvidiaSearch = {
        nvidia-cusolver-cu12 = [
          { name = "nvidia-cublas-cu12"; sub = "cublas"; }
          { name = "nvidia-cusparse-cu12"; sub = "cusparse"; }
          { name = "nvidia-nvjitlink-cu12"; sub = "nvjitlink"; }
        ];
        nvidia-cusparse-cu12 = [
          { name = "nvidia-nvjitlink-cu12"; sub = "nvjitlink"; }
        ];
        nvidia-cudnn-cu12 = [
          { name = "nvidia-cublas-cu12"; sub = "cublas"; }
        ];
        torch = cudaWheelLibs;
        triton = cudaWheelLibs;
        cupy-cuda12x = cudaWheelLibs;
      };
      cupyOptionalMissing = [
        "libcutensor.so.2"
        "libcutensorMg.so.2"
      ];
      patchNvidia = name: prev.${name}.overrideAttrs (old: {
        autoPatchelfIgnoreMissingDeps =
          (old.autoPatchelfIgnoreMissingDeps or [ ])
          ++ cudaDriverMissing
          ++ (if name == "cupy-cuda12x" then cupyOptionalMissing else [ ]);
        preFixup = (old.preFixup or "") + addNvidiaLibSearch (nvidiaSearch.${name} or [ ]);
      });
      nvidiaWheelNames = [
        "cuda-bindings"
        "cuda-toolkit"
        "nvidia-cublas-cu12"
        "nvidia-cuda-cupti-cu12"
        "nvidia-cuda-nvrtc-cu12"
        "nvidia-cuda-runtime-cu12"
        "nvidia-cudnn-cu12"
        "nvidia-cufft-cu12"
        "nvidia-cufile-cu12"
        "nvidia-curand-cu12"
        "nvidia-cusolver-cu12"
        "nvidia-cusparse-cu12"
        "nvidia-cusparselt-cu12"
        "nvidia-nccl-cu12"
        "nvidia-nvjitlink-cu12"
        "nvidia-nvshmem-cu12"
        "nvidia-nvtx-cu12"
        "torch"
        "triton"
        "cupy-cuda12x"
      ];
      nvidiaIgnores = lib.listToAttrs (
        map
          (name: { inherit name; value = patchNvidia name; })
          nvidiaWheelNames
      );
    in
    withBuildSystems // nvidiaIgnores // {
      ${projectName} = prev.${projectName}.overrideAttrs (old: {
        src = projectSrc;
      });
    };
  pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
    inherit python;
  }).overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      overlay
      pyprojectOverrides
    ]
  );
in
pythonSet.mkVirtualEnv "gpu-runtime-env" {
  ${projectName} = [ "gnn" ];
}
