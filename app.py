
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "TN5000 App Running Successfully!"

if __name__ == '__main__':
    app.run(debug=True)
