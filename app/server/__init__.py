from flask import Flask, render_template, request
import os, sys
from api import api


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/load_files", methods=["POST"])
def load_files():
    print(request.form["sc_path"])
    print(request.form["bulk_path"])

    sadata, badata  = api.load_files(request.form["sc_path"], request.form["bulk_path"])
    return render_template("load_files.html")


    
