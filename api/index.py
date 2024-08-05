from flask import Flask, request, jsonify
import numpy as np
import os
# from flask_cors import CORS
# CORS(app)

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, I added importations'

@app.route('/about')
def about():
    return 'About'