from http.client import responses
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import telebot

from tensorflow.python.framework import ops

with open("intents.json") as file:
    data = json.load(file)

try:
    with  open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

training = numpy.array(training)
output = numpy.array(output)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 200)
net = tflearn.fully_connected(net, 200)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=200, batch_size=8, show_metric=True)
model.save("model.tflearn")


bot_token = '5312873097:AAFULa4JC_wTEe41vEMZjiE3BJAUkuZXNKE'
bot = telebot.TeleBot(token=bot_token)

pesan = ''
jumlah_pesanan = 0
item = 'clear'
tagg = 'bot'



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

@bot.message_handler()
def chat(message):

    global pesan, jumlah_pesanan, item, tagg
    inp = message.text
    print(item)

    
    for menu in data["menu"]:
        if menu["tag"] == item:
            try:
                num = int(inp)
            except ValueError:
                bot.reply_to(message, "Jumlah porsi yang kamu masukkan salah")
                bot.reply_to(message, "Silahkan masukkan kembali porsi yang kamu butuhkan")
                tagg = ''
                continue
            else:
                pesanan = num * int(menu["harga"])
                pesan = pesan + item + ' (' + inp + ' x Rp' + menu["harga"] + ') = Rp' + str(pesanan) + "\n"
                jumlah_pesanan = jumlah_pesanan + pesanan
                bot.reply_to(message, "Rp" + str(pesanan))
                bot.reply_to(message, "Silahkan masukkan menu lain jika kamu ingin menambahkan pesanan atau ketik 'Tidak' jika sudah selesai")
                item = 'clear'
                tagg = ''
            

        if menu['tag'] == inp:
            jumlah = ["Kamu butuh berapa porsi?", "Berapa porsi?", "Perlu berapa porsi?"]
            bot.reply_to(message, random.choice(jumlah))
            item = inp
            tagg = ''

    if inp == 'Tidak' or inp == 'tidak':
        pesan = ["\nPesanan kamu :\n" + pesan + "\nTotal biaya yang harus kamu bayar adalah Rp" + str(jumlah_pesanan) + "\n\nApabila sudah benar silahkan hubungi @shafwarijal untuk konfirmasi dan pembayaran." + "\n\nKetik 'Selesai' untuk mengakhiri percakapan ini."]
        bot.reply_to(message, pesan)
        tagg = ''

    if inp == 'Selesai' or inp == 'selesai':
        pesan2 = ["Terima kasih sudah melakukan pemesanan di Ayam Geprek Moen Moen."]
        bot.reply_to(message, pesan2)
        pesan3 = ["Sampai bertemu kembali <3", "Sering-sering berkunjung ya!", "Kamu jangan terlalu lama menghilang ya!"]
        bot.reply_to(message, pesan3)
        pesan = ''
        jumlah_pesanan = 0
        item = 'clear'
        tagg = ''
        

    if tagg == "bot":
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            bot.reply_to(message, random.choice(responses))
            tagg = 'bot'
        else:
            bot.reply_to(message, "Maaf, kami tidak mengetahui yang kamu maksud.\nSilahkan coba lagi")
            tagg = 'bot'
    elif tagg == "":
        tagg = 'bot'
        

bot.polling()