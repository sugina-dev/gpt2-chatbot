from interact_mmi import Talk
import os

t = Talk()

TOKEN = os.environ['BOT_TOKEN']

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher

def start(update, context):
	chat_id = update.effective_chat.id
	context.bot.send_message(chat_id=chat_id, text='''我是一個 AI。我還不是很會説話，請多多關照。
我的開發者們能被發現在 [這裏](https://github.com/sugina-dev/gpt2-chinese-chatbot)。
要讓我忘記和你聊過的天的話，使用 /destroy 命令來達到吧。
下面開始跟你聊天喔。''', parse_mode='Markdown', disable_web_page_preview=True)
dispatcher.add_handler(CommandHandler('start', start))

def destroy(update, context):
	chat_id = update.effective_chat.id
	if t.remove_talk(chat_id):
		context.bot.send_message(chat_id=chat_id, text='銷毀已完成')
	else:
		context.bot.send_message(chat_id=chat_id, text='已無事可做')
dispatcher.add_handler(CommandHandler('destroy', destroy))

def chat(update, context):
	chat_id = update.effective_chat.id
	text = update.message.text
	reply = t.start_talk(chat_id, text)
	context.bot.send_message(chat_id=chat_id, text=reply)
dispatcher.add_handler(MessageHandler(Filters.text, chat))

updater.start_polling()
