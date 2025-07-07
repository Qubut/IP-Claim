{
  description = "Flake to manage Python workspace with UV and CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, uv2nix, pyproject-nix, pyproject-build-systems, ... }:
    let
      inherit (nixpkgs) lib;
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.6" ];
        };
      };

      workspace = uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = ./.;
      };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      python = pkgs.python312;
      hacks = pkgs.callPackage pyproject-nix.build.hacks { };

      pyprojectOverrides = final: prev: {
        antlr4-python3-runtime = prev.antlr4-python3-runtime.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
        });
        spacy = prev.spacy.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.setuptools ];
        });
        spacy-experimental = prev.spacy.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.setuptools ];
        });
      };

      pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
        inherit python;
      }).overrideScope (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
        pyprojectOverrides
      ]);

    in
    {
      packages.${system}.default = pythonSet.mkVirtualEnv "ip_claim-prod" workspace.deps.default;

      devShells.${system}.default =
        let
          virtualenv = pythonSet.mkVirtualEnv "ip_claim-dev" workspace.deps.all;
        in
        pkgs.mkShell {
          NIX_BUILD_CORES = 4;
          NIX_REMOTE = "daemon";
          packages = with pkgs; [
            virtualenv
            uv
            bzip2
            python311
            jupyter
            python311Packages.ipykernel
          ];

          env = {
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];
            UV_NO_SYNC = "1";
            # UV_PYTHON = "${virtualenv}/bin/python";
          };

          shellHook = ''
          unset PYTHONPATH
          uv venv .venv
          source .venv/bin/activate
          uv pip install -e .
          '';
        };
    };
}
