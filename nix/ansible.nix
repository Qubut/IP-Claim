{ config, ... }:

{
  languages.ansible.enable = true;

  # Galaxy collections from ops/ansible/requirements.yml install on devenv shell
  # entry when the requirements file changes (no manual ansible-galaxy step).
  tasks."ip-claim:ansible-galaxy" = {
    exec = ''
      ansible-galaxy collection install -r "${config.env.DEVENV_ROOT}/ops/ansible/requirements.yml"
    '';
    before = [ "devenv:enterShell" ];
    execIfModified = [ "ops/ansible/requirements.yml" ];
  };
}
