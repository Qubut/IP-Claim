
import openai
import sglang as sgl
from sglang.utils import launch_server_cmd, print_highlight, terminate_process, wait_for_server

from utils import load_config

if __name__ == '__main__':
    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    config = load_config('./config/sglang.yml')
    cmd_args = ' '.join([
    f'--{k} {v}' if not isinstance(v, bool) else f'--{k}' for k, v in config['cmd'].items()
])
    cmd = f'python -m sglang.launch_server {cmd_args}'
    server_process, port = launch_server_cmd(cmd)

    wait_for_server(f'http://localhost:{port}')
    client = openai.Client(base_url=f'http://127.0.0.1:{port}/v1', api_key='None')
