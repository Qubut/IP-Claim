{ pkgs ? import <nixpkgs> { }
, pythonEnvPath
, runtimeRootPath ? null
, name ? "localhost/ipclaim-ssv-gpu-base"
, tag ? "nix"
}:
# Stable GPU runtime image: uv2nix virtualenv plus runtime-root only. No src
# or configs. streamLayeredImage yields a script; hydrate once into the isolated
# Podman engine; job overlays compose on top without re-streaming torch.
let
  pythonEnv = builtins.storePath pythonEnvPath;
  runtimeRoot =
    if runtimeRootPath == null
    then import ./runtime-root.nix { inherit pkgs; }
    else builtins.storePath runtimeRootPath;
  libPath = pkgs.lib.makeLibraryPath [
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
    pkgs.libsndfile
  ];
in
pkgs.dockerTools.streamLayeredImage {
  inherit name tag;
  contents = [ runtimeRoot pythonEnv ];
  extraCommands = ''
    mkdir -p app tmp
    chmod 1777 tmp
    ln -s ${pythonEnv} app/.venv
  '';
  config = {
    Env = [
      "PATH=/app/.venv/bin:/bin:/usr/bin"
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "LD_LIBRARY_PATH=${libPath}"
    ];
    WorkingDir = "/app";
  };
}
