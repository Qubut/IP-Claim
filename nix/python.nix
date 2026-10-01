{ pkgs, lib, config, ... }:
let
  pythonVersion = "python312";
  pythonVersionDot = lib.pipe pythonVersion [
    (lib.strings.removePrefix "python")
    (v: "${lib.strings.substring 0 1 v}.${lib.strings.substring 1 2 v}")
  ];
  pythonPackage = pkgs.${pythonVersion};
  venvDir = "${config.env.DEVENV_ROOT}/.devenv/state/venv";
  srcPath = "${config.env.DEVENV_ROOT}/src";
  extraPaths = [ srcPath ];
in
{
  env = {
    PYTEST_ADDOPTS = "--verbose";  # Default pytest options
    JUPYTER_CONFIG_DIR = "${config.env.DEVENV_ROOT}/.jupyter";
    JUPYTER_DATA_DIR = "${config.env.DEVENV_ROOT}/.jupyter";
    JUPYTER_RUNTIME_DIR = "${config.env.DEVENV_ROOT}/.jupyter/runtime";
    CC = "${pkgs.stdenv.cc}/bin/cc";
    CXX = "${pkgs.stdenv.cc}/bin/c++";
    UV_PYTHON = "${pythonPackage}/bin/python";
    # Large wheels (torch, CUDA libs) exhaust a small /tmp on shared hosts.
    # Keep extract and uv cache on the project disk (.cache/ is gitignored).
    TMPDIR = "${config.env.DEVENV_ROOT}/.cache/tmp";
    UV_CACHE_DIR = "${config.env.DEVENV_ROOT}/.cache/uv";
    XDG_CACHE_HOME = "${config.env.DEVENV_ROOT}/.cache";
    HF_HOME = "${config.env.DEVENV_ROOT}/.cache/huggingface";
  };
  languages.python = {
    enable = true;
    package = pythonPackage;
    uv.enable = true;
    uv.sync.enable = true;
    uv.sync.groups = [ "dev" "test" ];
    venv.enable = true;
  };
  files.".python-version".text = pythonVersionDot;
  files."pyrightconfig.json".text = builtins.toJSON {
    include = [ "src" "tests" "experiments" ];
    extraPaths = extraPaths;
    exclude = [ ".devenv" ".cache" ];
    pythonVersion = pythonVersionDot;
    venvPath = "${config.env.DEVENV_ROOT}/.devenv/state";
    venv = "venv";
  };
  files.".vscode/settings.json".text = builtins.toJSON {
    "nixEnvSelector.nixFile" = "\${workspaceFolder}/.devenv.flake.nix";
    "nixEnvSelector.useFlakes" = false;
    "python.analysis.diagnosticMode" = "workspace";
    "python.defaultInterpreterPath" = "${venvDir}/bin/python";
    "python.pythonPath" = "${venvDir}/bin/python";
    "python.analysis.extraPaths" = extraPaths;
    "python.autoComplete.extraPaths" = extraPaths;
    "pyright.analysis.extraPaths" = extraPaths;
    "basedpyright.analysis.extraPaths" = extraPaths;
    "basedpyright.analysis.diagnosticMode" = "workspace";
    "python.languageServer" = "None";
    "files.exclude" = {
      "**/*.vo" = true;
      "**/*.vok" = true;
      "**/*.vos" = true;
      "**/*.aux" = true;
      "**/*.glob" = true;
      "**/.git" = true;
      "**/.svn" = true;
      "**/.hg" = true;
      "**/.DS_Store" = true;
      "**/Thumbs.db" = true;
    };
  };
}
