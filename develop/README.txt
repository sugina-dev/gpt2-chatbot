🔶️ 預訓練模型

dialogue_model
Google Drive：https://drive.google.com/drive/folders/1Ogz3eapvtvdY4VUcY9AEwMbNRivLKhri?usp=sharing
使用闲聊语料训练了 40 个 epoch，最终 loss 在 2.0 左右，继续训练的话，loss 应该还能继续下降。

mmi_model
Google Drive：https://drive.google.com/drive/folders/1oWgKXP6VG_sT_2VMrm0xL4uOqfYwzgUP?usp=sharing
以 dialogue_model 作为预训练模型，使用上述闲聊语料，训练了 40 个 epoch，最终 loss 在 1.8-2.2 之间，继续训练的话，loss 也能继续下降

The total size is 627 MB.

🔶️ Prerequisites

Download the above files:

develop
├───dialogue_model
│       config.json
│       pytorch_model.bin
└───mmi_model
        config.json
        pytorch_model.bin

🔶️ Run

$ docker build -t gpt2-chatbot . --no-cache
$ docker run -d -e BOT_TOKEN='YOUR_BOT_TOKEN' --name=my-gpt2-chatbot gpt2-chatbot

Set the environment variable BOT_TOKEN to bot token.

🔶️ Troubleshooting

1. Use pip mirror when docker build:

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

2. Start bot with proxy:

REQUEST_KWARGS = { 'proxy_url': 'socks5h://localhost:1080' }
updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)

🔶️ Test

$ python interact_mmi.py --no_cuda
