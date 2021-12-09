from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    # return "<h1>Index Page</h1>"
    return render_template('index.html',  title='Home')


# @app.route("/hello")
# def hello():
#     return "<h1>Hello World</h1>"
    