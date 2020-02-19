# GPT2 實作中文閒聊機械人

## 项目结构
- config:存放GPT2模型的参数的配置文件
- data
    - train.txt:默认的原始训练集文件，存放闲聊语料 
    - train_tokenized.txt:对原始训练语料进行顺序tokenize之后的文件，用于dialogue model的训练
    - train_mmi_tokenized.txt:对原始训练语料进行逆序tokenize之后的文件，用于mmi model的训练
- dialogue_model:存放对话生成的模型
- mmi_model:存放MMI模型(maximum mutual information scoring function)，用于预测P(Source|response)
- sample:存放人机闲聊生成的历史聊天记录
- vocabulary:存放GPT2模型的字典
- train.py:训练代码
- interact.py:人机交互代码

## 思想

- 解码器的逻辑使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文[The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- 根据微软的[DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)的思想，在项目中添加了互信息。训练了两个模型:Dialogue Model与MMI Model(maximum mutual information scoring function)。首先使用Dialogue Model生成多个候选response，然后使用MMI Model从候选response中，选取loss最小的作为最终的response
- 代码中给出了许多详细的中文注释，方便大家更好地理解代码(能力有限，可能有些代码或注释有误，望大家不吝赐教)

## 用灋

### 下載預訓練模型

闲聊语料大小为67M，包含50w个多轮对话。使用该语料训练了两个模型dialogue_model与mmi_model

|模型 | 百度网盘 |GoogleDrive |模型描述|
|---------|--------|--------|--------|
|dialogue_model | [百度网盘【提取码:osi6】](https://pan.baidu.com/s/1qDZ24VKLBU9GKARX9Ev65g) | [GoogleDrive](https://drive.google.com/drive/folders/1Ogz3eapvtvdY4VUcY9AEwMbNRivLKhri?usp=sharing) |使用闲聊语料训练了40个epoch，最终loss在2.0左右，继续训练的话，loss应该还能继续下降。|
|mmi_model | [百度网盘【提取码:1j88】](https://pan.baidu.com/s/1ubXGuEvY8KmwEjIVTJVLww) | [GoogleDrive](https://drive.google.com/drive/folders/1oWgKXP6VG_sT_2VMrm0xL4uOqfYwzgUP?usp=sharing) |以dialogue_model作为预训练模型，使用上述闲聊语料，训练了40个epoch，最终loss在1.8-2.2之间，继续训练的话，loss也能继续下降。|

### 運行

把下载好的模型文件夹dialogue_model与mmi_model放在项目根目录下(否则需要通过--dialogue_model_path与--mmi_model_path参数指定对应模型的路径)，执行如下命令:

#### 仅使用dialogue_model进行生成

``` bash
python interact.py --no_cuda(使用默认参数，不使用GPU。由于闲聊对话生成的内容长度不是很长，因此生成部分在CPU上跑速度也挺快的)
或
python interact.py --no_cuda --dialogue_model_path path_to_dialogue_model --max_history_len 5(自定义--max_history_len参数，即对话历史的长度)
或
python interact.py --no_cuda --dialogue_model_path path_to_dialogue_model --max_history_len 5 --topp 0.8 --topk 0(--topp为0到1之间的小数，用于调用Nucleus Sampling)
或
python interact.py --no_cuda --max_history_len 5 --topk 8(未指定--dialogue_model_path参数，默认为dialogue_model)
```

输入Ctrl+Z结束对话之后，聊天记录将保存到sample目录下的sample.txt文件中

#### 使用dialogue_model生成多个候选response，然后使用mmi_model选取互信息loss最小的response

interact_mmi.py的用法与interact.py类似
``` bash
python interact_mmi.py --no_cuda(使用默认的model路径)
或
python interact_mmi.py --no_cuda --batch_size 5(指定生成候选response的个数)
或
python interact_mmi.py --no_cuda --debug(debug模式，可以看到生成的所有候选response及其通过mmi_model的loss)
或
python interact_mmi.py --no_cuda --dialogue_model_path path_to_dialogue_model --mmi_model_path path_to_mmi_model(自定义模型路径)
```
输入Ctrl+Z结束对话之后，聊天记录将保存到sample目录下的mmi_samples.txt文件中

更多的参数介绍，可直接看interact.py和interact_mmi.py中的setup_train_args()函数中的参数说明

### interact.py与interact_mmi.py的参数

执行interact.py时，可以尝试通过调整topk、topp、repetition_penalty、max_history_len等参数，调整生成的效果。详细的参数描述可以查看interact.py的set_interact_args()函数

## 模型参数(详见config/model_config_dialogue_small.json文件)
- initializer_range: 0.02
- layer_norm_epsilon: 1e-05
- n_ctx: 300
- n_embd: 768
- n_head: 12
- n_layer: 10
- n_positions: 300
- vocab_size: 13317

## Dialogue Model
Dialogue Model是基于GPT2模型的生成模型，对每条训练数据进行"顺序"拼接，然后将其输入到网络中，进行训练(此处的"顺序"是相对于MMI Model的"逆序")

例如存在如下多轮闲聊训练数据,在训练Dialogue Model时，将上述训练数据进行如下拼接:**"[CLS]想看你的美照[SEP]亲我一口就给你看[SEP]我亲两口[SEP]讨厌人家拿小拳拳捶你胸口[SEP]"**。然后将上述拼接结果作为Dialogue Model的输入，对模型进行训练
```
想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口
```

## MMI Model

MMI Model的思想基于微软的论文[DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)

MMI Model也是一个基于GPT2的生成模型，将每条训练数据进行"逆序"拼接,然后输入到网络中。该模型主要用于计算Dialogue Model生成的所有候选response相对于dialogue history的loss。

训练时，将一条训练语料进行逆序拼接，如 **"[CLS]讨厌人家拿小拳拳捶你胸口[SEP]我亲两口[SEP]亲我一口就给你看[SEP]想看你的美照[SEP]"**，并作为MMI Model的输入进行训练
```
想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口
```

## response生成步骤
- 假设当前dialogue history=["你好","你好呀","你在干嘛呢"]
- 首先使用Dialogue Model根据dialogue history生成n个候选response:["在看电视","我在上课啊","人家在想你啊","我不知道"]
- 使用MMI Model将每个候选response分别与dialogue history进行逆序拼接，如 **"[CLS]在看电视[SEP]你在干嘛呢[SEP]你好呀[SEP]你好[SEP]"**
- 将上述拼接结果作为MMI Model的输入，计算每个response的loss
- 选择loss最小的response作为最终的结果进行回复


## 闲聊语料分享
|中文闲聊语料 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料 | [百度网盘【提取码:jk8d】](https://pan.baidu.com/s/1mkP59GyF9CZ8_r1F864GEQ) 或 [GoogleDrive](https://drive.google.com/file/d/1nEuew_KNpTMbyy7BO4c8bXMXN351RCPp/view?usp=sharing) |由作者[GaoQ1](https://github.com/GaoQ1)提供的比较高质量的闲聊数据集，整理出了50w个多轮对话的语料|

50w中文闲聊语料的内容样例如下:
```
谢谢你所做的一切
你开心就好
开心
嗯因为你的心里只有学习
某某某，还有你
这个某某某用的好

你们宿舍都是这么厉害的人吗
眼睛特别搞笑这土也不好捏但就是觉得挺可爱
特别可爱啊

今天好点了吗？
一天比一天严重
吃药不管用，去打一针。别拖着
```

## 训练模型
在项目根目录下创建data文件夹，将原始训练语料命名为train.txt，存放在该目录下。train.txt的格式如下，每段闲聊之间间隔一行。

训练模型：
``` bash
# 若是训练mmi_model则需要指定--train_mmi参数；若是训练dialogue_model，则不需要指定--train_mmi参数

#训练dialogue_model
python train.py --epochs 30 --batch_size 8 --device 0,1 --raw(若要对原始训练语料进行tokenize，则要指定--raw参数。若要用GPU训练，则通过--device指定GPU)
或
python train.py --epochs 30 --batch_size 8 --no_cuda --raw(指定--no_cuda参数，则使用CPU训练，速度要慢得多)
或
python train.py --epochs 30 --batch_size 8 --no_cuda(若已经对原始语料进行tokenize，可以不用指定--raw，避免重复tokenize，节约时间)

#训练mmi_model,要指定--train_mmi参数
python train.py --epochs 30 --batch_size 8 --device 0,1 --raw --train_mmi(对原始训练语料进行逆序拼接，tokenize，并且训练mmi_model)
或
python train.py --epochs 30 --batch_size 8 --device 0,1 --train_mmi(若已经对原始训练语料tokenize，则直接训练mmi_model)
或
python train.py --epochs 30 --batch_size 8 --device 0,1 --train_mmi --pretrained_model path_to_pretrained_model(在与训练模型基础上继续训练)
```
更多的参数介绍，可直接看train.py中的setup_train_args()函数中的参数说明

## 不足之处

没有对chatbot生成的response的内容进行充分检测，有时会生成一些敏感、略带玩笑性质的辱骂内容

## Reference
- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)

