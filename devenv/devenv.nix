{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [
    git
    process-compose
    honcho
    glibc
    glibc.static
    podman-compose
    secretspec
    pkg-config
  ];

  process.manager.implementation = "honcho";

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  enterShell = ''
    hello
    git --version
  '';


  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
  git-hooks.enable = true;
  git-hooks.hooks.shellcheck.enable = true;
}
