{ pkgs ? import <nixpkgs> { }
, pythonEnvPath
, runtimeRootPath ? null
, srcPath ? null
, configsPath ? null
, name ? "localhost/ipclaim-ssv-gpu-runtime"
, tag ? "nix"
}:
# Back-compat entry: when src/configs are omitted this is the stable base
# stream; when both are set it builds a single combined image. Prefer
# base.nix plus overlay.Containerfile so a warm base is not
# re-transferred every job.
let
  base = import ./base.nix {
    inherit pkgs pythonEnvPath runtimeRootPath name tag;
  };
in
if srcPath == null && configsPath == null then
  base
else
  let
    pythonEnv = builtins.storePath pythonEnvPath;
    runtimeRoot =
      if runtimeRootPath == null
      then import ./runtime-root.nix { inherit pkgs; }
      else builtins.storePath runtimeRootPath;
    src = builtins.path { path = srcPath; name = "src"; };
    configs = builtins.path { path = configsPath; name = "configs"; };
    libPath = pkgs.lib.makeLibraryPath [
      pkgs.zlib
      pkgs.stdenv.cc.cc.lib
    ];
  in
  pkgs.dockerTools.streamLayeredImage {
    inherit name tag;
    contents = [ runtimeRoot pythonEnv src configs ];
    extraCommands = ''
      mkdir -p app tmp
      chmod 1777 tmp
      ln -s ${pythonEnv} app/.venv
      ln -s ${src} app/src
      ln -s ${configs} app/configs
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
