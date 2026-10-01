{ pkgs ? import <nixpkgs> { } }:
# Shared FHS-ish runtime bits for the GPU base image (shell, codecs, CA bundle,
# linker libs). Realized as its own store path so it can be nix-copied to the
# remote multi-user store independently of the volatile src/config overlay.
pkgs.buildEnv {
  name = "gpu-runtime-root";
  paths = [
    pkgs.bashInteractive
    pkgs.coreutils
    pkgs.curl
    pkgs.cacert
    pkgs.gitMinimal
    pkgs.gnugrep
    pkgs.gnused
    pkgs.findutils
    pkgs.gawk
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
    pkgs.dockerTools.usrBinEnv
    pkgs.dockerTools.binSh
    pkgs.dockerTools.fakeNss
  ];
  pathsToLink = [ "/bin" "/usr" "/etc" "/lib" ];
}
