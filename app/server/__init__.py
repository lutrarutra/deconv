from flask import Flask, render_template
import os, sys
from deconv.api import api


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/load_files", methods=["POST"])
def load_files():
    sadata, badata  = api.load_files()
    return render_template("load_files.html")


    
