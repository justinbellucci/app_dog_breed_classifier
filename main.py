from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def home_view():
    # return "<h1>Hello, World!</h1>"
    return render_template('home.html',  title='Home')