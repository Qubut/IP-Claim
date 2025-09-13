{ pkgs, lib, config, ... }:

{
  devcontainer.enable = true;
env = {
    PYTEST_ADDOPTS = "--verbose";  # Default pytest options
    UV_PYTHON = "${config.languages.python.package}/bin/python";
  };

  packages = with pkgs; [
    jupyter
    doppler
    gcc
    libgcc
    gnumake
    cmake
    extra-cmake-modules
    uv
    ruff
    black
    isort
    python311Packages.pip
    zip
  ];

  languages.python = {
    enable = true;
    package = pkgs.python311;
    uv.enable = true;
    venv.enable = true;
  };

  pre-commit.hooks = {
    black.enable = true;
    # ruff.enable = true;
    mypy.enable = true;
  };

  dotenv.disableHint = true;
  cachix.enable = false;

  enterShell = ''
    echo "Python dev environment ready"
  '';
}
