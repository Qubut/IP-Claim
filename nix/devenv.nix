{ pkgs, ... }:

# Server-side ops tooling imported from devenv.yaml. Python ML stack lives in
# the repo-root devenv.nix; this module only adds container/remote helpers.
{
  packages = with pkgs; [
    podman-compose
  ];

  git-hooks.hooks.shellcheck.enable = true;
}
