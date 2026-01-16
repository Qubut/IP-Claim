{ pkgs, lib, config, ... }:

let
  pythonVersion = "python312";
  pythonVersionDot = lib.pipe pythonVersion [
    (lib.strings.removePrefix "python")
    (v: "${lib.strings.substring 0 1 v}.${lib.strings.substring 1 2 v}")
  ];
  pythonPackage = pkgs.${pythonVersion};
  venvPath = "${config.env.DEVENV_ROOT}/.devenv/state/venv";
  in
{
  env = {
    PYTEST_ADDOPTS = "--verbose";  # Default pytest options
    UV_PYTHON = "${config.languages.python.package}/bin/python";
    PYTHONPATH = "${venvPath}/lib/${pythonVersion}/site-packages";
    JUPYTER_CONFIG_DIR = "${config.env.DEVENV_ROOT}/.jupyter";
    JUPYTER_DATA_DIR = "${config.env.DEVENV_ROOT}/.jupyter";
    JUPYTER_RUNTIME_DIR = "${config.env.DEVENV_ROOT}/.jupyter/runtime";
    CC = "${pkgs.stdenv.cc}/bin/cc";
    CXX = "${pkgs.stdenv.cc}/bin/c++";
    LDFLAGS = "-L${pkgs.glibc}/lib";
  };

  languages.python = {
    enable = true;
    package = pythonPackage;
    uv.enable = true;
    venv.enable = true;
    uv.sync.enable = true;
  };

  processes.jupyter.exec = "uv run jupyter server --ip=0.0.0.0 --port=8888 --no-browser";

  files."pyrightconfig.json".text = builtins.toJSON {
    venvPath = config.env.DEVENV_STATE;
    venv = "venv";
    extraPaths = [ "${venvPath}/lib/${pythonVersion}/site-packages" ];
    useLibraryCodeForTypes = true;
    analysis = {
      autoSearchPaths = true;
      diagnosticMode = "workspace";
      useLibraryCodeForTypes = true;
    };
  };
   files.".python-version".text="${pythonVersionDot}";

  enterShell = ''
    echo "Python dev environment ready"
  '';
}
