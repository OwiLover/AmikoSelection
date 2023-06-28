from numpy import NaN
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, ReplyMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters, ConversationHandler
import sqlite3
import functions as fs
import time
import threading

con = sqlite3.connect('Amiko.db', check_same_thread=False)
cur=con.cursor()

reply_keyboard = [['/Login']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False, resize_keyboard=False)

def start (bot, update):
    bot.message.reply_text(f'Я Бот Амико! Создан для упрощения работы ассистентов в наших магазинах, если ты являешься сотрудником, то нажми /Login!', reply_markup=markup)

def login (bot, update):
    user_id = bot.message.from_user.id
    num=bot.message.from_user.username
    check = fs.addUser(user_id, num)
    if check == "Старый Пользователь":
        reply_keyboard = [['Начнём!'],["/cancel"]]
        bot.message.reply_text(f'Приветик, {bot.effective_user.first_name}, когда закончишь, не забудь написать /cancel',reply_markup=ReplyKeyboardMarkup(reply_keyboard,one_time_keyboard=False,resize_keyboard=False),)
        cur.execute("DELETE FROM Buffer WHERE UserId=?",(user_id,))
        cur.execute("INSERT INTO Buffer (UserID, isWaiting) VALUES(?,?)", (int(user_id),0))
        con.commit();
        return 1
    else:
        bot.message.reply_text(f'У Вас нет доступа')
        return ConversationHandler.END

def hello(update, context: CallbackContext) -> None:
    update.message.reply_text(f'Приветик {update.effective_user.first_name}')

def ping(update, context: CallbackContext) :
    update.message.reply_text(f'Pong')
    print("Pong")

def Waiting(update, curr):
    user_id = update.message.from_user.id
    while curr is None:
        for value in cur.execute("SELECT * FROM Buffer WHERE UserId =?;", (user_id,)):
            curr = value[0]
        time.sleep(1)
        print(curr)
    reply_keyboard = [['Следующий!'],["/cancel"]]
    update.message.reply_text(
        'Подойдите в отдел '+str(curr),
        reply_markup=ReplyKeyboardMarkup(reply_keyboard,one_time_keyboard=False,resize_keyboard=False),
    )
    cur.execute("UPDATE Buffer SET isWaiting=? WHERE UserId=?",(0,user_id))
    con.commit();

def pswd(update, context: CallbackContext):
    user_id = update.message.from_user.id
    status = 0
    for value in cur.execute("SELECT * FROM Buffer WHERE UserId =?;", (user_id,)):
        status = value[2]
    if status == 0:
        update.message.reply_text(
            'В ожидании посетителей',
            reply_markup=ReplyKeyboardRemove(),
        )
        cur.execute("UPDATE Buffer SET isWaiting=? WHERE UserId=?",(1,user_id))
        con.commit();
        cur.execute("UPDATE Buffer SET Name=? WHERE UserId=?",(None,user_id))
        con.commit();
        curr = None
        threading.Thread(target=Waiting, args=[update,curr]).start()
    else: update.message.reply_text(
            'Вы уже ждёте посетителей',
            reply_markup=ReplyKeyboardRemove(),
        )
    return 1



def cancel (update, context: CallbackContext):
    update.message.reply_text(f'Как скажете, возвращаю Вас в главное меню', reply_markup=markup)
    return ConversationHandler.END

conv_handler = ConversationHandler(
        entry_points=[CommandHandler('login', login)],
        
        states={
           1: [MessageHandler(Filters.text & (~ Filters.command),  pswd, pass_user_data=True)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )


updater = Updater('YOUR_TOKEN')

dp = updater.dispatcher

dp.add_handler(conv_handler)
dp.add_handler(CommandHandler('hello', hello))
dp.add_handler(CommandHandler('start', start))
dp.add_handler(CommandHandler('ping', ping))

updater.start_polling()
updater.idle()
con.close()

