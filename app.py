from flask import Flask, render_template, request

from sentiment import pipeline


app = Flask(__name__)

@app.route('/' , methods = ['GET' , 'POST'])
def home():

    if request.method == 'POST':
        query = request.form['query']

        language = request.form['lang']

        page_size = request.form['articles']


        sentiment = pipeline(query, language, page_size)

    else:
        sentiment = {'values': [], 'labels': [], 'colors': []}
        
    return render_template('index.html', sentiment = sentiment)

app.run(host = '0.0.0.0', port = 8000, debug = True)
