{ pkgs, inputs, config, lib, ... }:
{
  env = {
    GOPATH = lib.mkDefault "${config.env.DEVENV_ROOT}/.go";
    GOBIN = lib.mkDefault "${config.env.DEVENV_ROOT}/.go/bin";
    GOMODCACHE = lib.mkDefault "${config.env.DEVENV_ROOT}/.go/pkg/mod";
    GOTOOLCHAIN = "local";
  };

  languages.go = {
    enable = true;
    package = pkgs.go;
  };

  cachix.enable = false;

  packages = with pkgs; [
    jupyter
    go
    go-tools
    gopls
    air
    gofumpt
    golangci-lint
    goreleaser
    golines
  ];

  files.".vscode/settings.json".text = ''
    {
      "go.goroot": "${pkgs.go}/share/go",
      "go.alternateTools": {
        "go": "${pkgs.go}/bin/go",
        "gopls": "${pkgs.gopls}/bin/gopls",
        "staticcheck": "${pkgs.go-tools}/bin/staticcheck"
      },
      "go.toolsEnvVars": {
        "GOPATH": "${config.env.DEVENV_ROOT}/.go",
        "GOMODCACHE": "${config.env.DEVENV_ROOT}/.go/pkg/mod"
      },
      "go.toolsManagement.autoUpdate": false,
      "go.lintTool": "staticcheck"
    }
  '';

  enterShell = ''
    export PATH=$PATH:$GOBIN;
    mkdir -p $GOPATH $GOBIN $GOMODCACHE  # Ensure directories exist on shell entry
  '';
}
