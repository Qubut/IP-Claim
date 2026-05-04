{ pkgs, inputs, config, lib, ... }:
{
  env = {
    EPO_PROCESSOR_DIR = config.env.DEVENV_ROOT;
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

  # Convenience scripts (run via `devenv shell <name>` or just `<name>` once
  # the shell is entered). They wrap the Makefile and the produced binary so
  # contributors don't have to remember the cmd/epo_processor path quirk.
  scripts = {
    epo-build.exec = ''
      cd "$EPO_PROCESSOR_DIR" && make build
    '';
    epo-run.exec = ''
      cd "$EPO_PROCESSOR_DIR"
      [ -x bin/epo-processor ] || make build
      exec ./bin/epo-processor "$@"
    '';
    epo-process.exec = ''
      cd "$EPO_PROCESSOR_DIR"
      [ -x bin/epo-processor ] || make build
      exec ./bin/epo-processor process --config config/config.yaml "$@"
    '';
    epo-process-hupd.exec = ''
      cd "$EPO_PROCESSOR_DIR"
      [ -x bin/epo-processor ] || make build
      exec ./bin/epo-processor process-hupd --config config/config.yaml "$@"
    '';
    epo-analyze.exec = ''
      cd "$EPO_PROCESSOR_DIR"
      [ -x bin/epo-processor ] || make build
      exec ./bin/epo-processor analyze --config config/config.yaml "$@"
    '';
    epo-test.exec = ''
      cd "$EPO_PROCESSOR_DIR" && go test -race ./...
    '';
  };
}
