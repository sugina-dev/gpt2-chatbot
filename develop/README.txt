Usage:

Download the following files:

.
├───dialogue_model
│       config.json
│       pytorch_model.bin
│
└───mmi_model
        config.json
        pytorch_model.bin

The total size is 627 MB.

Set the environment variable BOT_TOKEN to bot token.

Run python interact_mmi.py --no_cuda.

Dockerfile pip mirror:

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

This set the pip mirror to aliyun.com

Run with proxy:

REQUEST_KWARGS = { 'proxy_url': 'socks5h://localhost:1080' }
updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)
