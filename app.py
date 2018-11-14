from flask import Flask
from flask import render_template, request, jsonify
import classify
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = json.loads(request.data)   
    genre, keyword = classify.predictText(data["movie_description"])
    response = {
        "genres" : genre,
        "keywords" : keyword
    }
    return jsonify(response)

