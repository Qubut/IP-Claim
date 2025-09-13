{ pkgs, inputs, ... }:

{
  languages.rust = {
    enable = true;
    channel = "nixpkgs";
    components = [
      "rustc"      # Rust compiler
      "cargo"      # Rust package manager
      "clippy"     # Linter for catching common mistakes
      "rustfmt"    # Code formatter
      "rust-analyzer" # Language server for IDE support
    ];
  };

  packages = with pkgs; [
    # Tools for debugging and performance
    rustup
    gdb         # Debugger
    lldb        # LLVM-based debugger
    valgrind    # Memory debugging
    # Tools for code quality
    cargo-audit # Security vulnerability scanner
    cargo-watch # File watcher for automatic rebuilds
    # Utilities
    pkg-config
    openssl
    libxmlxx
    # libclang
    # llvmPackages.libcxxClang
    llvmPackages_14.libclang
    llvmPackages_14.clang
    llvmPackages_14.llvm
    glibc
  ];

  # Setting environment variables
  env = {
    RUST_LOG = "debug";              # Enable debug logging
    RUST_BACKTRACE = "1";            # Enable backtraces for debugging
    OPENSSL_DIR = "${pkgs.openssl.out}"; # Point to OpenSSL for linking
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
    CARGO_HOME = "../../.cargo";
    # LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  # Ensure libclang's dependencies (like libLLVM) are discoverable at runtime
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.llvmPackages_14.llvm
    pkgs.llvmPackages_14.libclang.lib
  ];
  LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
  BINDGEN_EXTRA_CLANG_ARGS = ''
  $(< ${pkgs.stdenv.cc}/nix-support/libc-crt1-cflags) \
  $(< ${pkgs.stdenv.cc}/nix-support/libc-cflags) \
  $(< ${pkgs.stdenv.cc}/nix-support/cc-cflags) \
  $(< ${pkgs.stdenv.cc}/nix-support/libcxx-cxxflags) \
  ${pkgs.lib.optionalString pkgs.stdenv.cc.isClang "-idirafter ${pkgs.stdenv.cc.cc}/lib/clang/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/include"} \
  ${pkgs.lib.optionalString pkgs.stdenv.cc.isGNU "-isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.lib.getVersion pkgs.stdenv.cc.cc} \
  -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/${pkgs.stdenv.hostPlatform.config} \
  -idirafter ${pkgs.stdenv.cc.cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/include"}
'';
  };

  pre-commit.hooks = {
    rustfmt.enable = true;
    clippy.enable = true;
    cargo-check.enable = true;
  };

  enterShell = ''
    echo "Rust dev environment ready"
    echo "Rust version: $(rustc --version)"
    echo "Cargo version: $(cargo --version)"
    echo "Available tools: rustfmt, clippy, rust-analyzer, cargo-audit, cargo-watch"
  '';

  processes = {
    cargo-watch = {
      exec = "cargo watch -x run";   # Run the project with file watching
    };
  };
}
