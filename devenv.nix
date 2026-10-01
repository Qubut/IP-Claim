{ pkgs, inputs, lib, config, ... }:

{
  outputs.gpu-python-env = import ./nix/gpu-python.nix {
    inherit pkgs lib;
    inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
    workspaceRoot = ./.;
  };

  devcontainer.enable = true;
  env = {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.zlib
      pkgs.zstd
      pkgs.libffi
      pkgs.stdenv.cc.cc.lib
    ] + ":${config.env.DEVENV_ROOT}/.devenv/nvidia-driver-libs";
  };

  packages = with pkgs; [
    jupyter
    doppler
    secretspec
    gcc
    libgcc
    gnumake
    cmake
    extra-cmake-modules
    uv
    ruff
    black
    isort
    zip
    openssh
    sshpass
    zlib
    zstd
    libffi
    texliveFull
    ansible-builder
    ansible-navigator
  ];
  # After `nix-store --gc`, `.devenv/state/venv/.devenv_interpreter` can point at
  # a removed store path. devenv's virtualenv task then fails in `readlink -f` under
  # `set -e` before it reaches its rebuild branch (devenv#336).
  tasks."ip-claim:venv-gc-recover" = {
    before = [ "devenv:python:uv" ];
    exec = ''
      cache_root="$DEVENV_ROOT/.cache"
      mkdir -p "$cache_root/tmp" "$cache_root/uv" "$cache_root/huggingface"

      venv_path="$DEVENV_STATE/venv"
      marker="$venv_path/.devenv_interpreter"
      if [ -f "$marker" ]; then
        interp=$(cat "$marker")
        if [ -n "$interp" ] && [ ! -e "$interp" ]; then
          echo "Removing stale devenv venv (interpreter no longer in nix store): $interp"
          rm -rf "$venv_path"
        fi
      fi
    '';
  };

  git-hooks.hooks = {
    # ruff format owns formatting in this project, so the black hook is off:
    # its output differs from the configured ruff style and rewrites files.
    black.enable = false;
    ruff.enable = true;
    # mypy needs the project environment (its plugins live in the uv venv),
    # so the standalone hook binary cannot type-check this tree. Types are
    # checked with `devenv shell -- mypy src`.
    mypy.enable = false;
  };

  dotenv.enable = true;
  dotenv.disableHint = true;
  cachix.enable = false;

  enterShell = ''
    cache_root="${config.env.DEVENV_ROOT}/.cache"
    mkdir -p "$cache_root/tmp" "$cache_root/uv" "$cache_root/huggingface"
    export TMPDIR="$cache_root/tmp"
    export UV_CACHE_DIR="$cache_root/uv"
    export XDG_CACHE_HOME="$cache_root"
    export HF_HOME="$cache_root/huggingface"

    # Expose ONLY the host NVIDIA driver lib (libcuda.so.1) to the venv via a
    # dedicated symlink dir referenced by LD_LIBRARY_PATH above. Linking the whole
    # host lib dir would clobber nix's glibc, so we symlink just libcuda.* here.
    nvidia_libdir="${config.env.DEVENV_ROOT}/.devenv/nvidia-driver-libs"
    libcuda=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.* /run/opengl-driver/lib/libcuda.so.* 2>/dev/null | grep -v '\.so\.1$' | head -1)
    if [ -n "$libcuda" ]; then
      mkdir -p "$nvidia_libdir"
      ln -sf "$libcuda" "$nvidia_libdir/libcuda.so.1"
      ln -sf libcuda.so.1 "$nvidia_libdir/libcuda.so"
    fi

    echo "Python dev environment ready"

    # GPU server SSH env. Credentials come from .env via dotenv; never print
    # SSH_PASS. Map the jump host from TEXTLESS_GPU_JUMP_HOST so no site alias
    # is hard-coded in version control.
    if [ -n "''${SSH_PASS:-}" ]; then
      export SSHPASS="''${SSH_PASS}"
      unset SSH_PASS
    fi
    export TEXTLESS_SERVER="''${TEXTLESS_SERVER:-''${TEXTLESS_GPU_SERVER:-}}"
    export TEXTLESS_SERVER_USER="''${TEXTLESS_SERVER_USER:-''${TEXTLESS_GPU_USER:-}}"
    if [ -n "''${TEXTLESS_GPU_JUMP_HOST:-}" ]; then
      export TEXTLESS_SERVER_SSH_ARGS="-o ProxyCommand=\"sshpass -e ssh -o ConnectTimeout=8 -W %h:%p ''${TEXTLESS_GPU_JUMP_HOST}\""
    fi
  '';
}
