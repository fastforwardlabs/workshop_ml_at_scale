from flask import Flask,send_from_directory,request
import logging
from pandas.io.json import dumps as jsonify
import os
import random
from IPython.display import Javascript,HTML


# This reduces the the output to the 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app= Flask(__name__,static_url_path='')
@app.route('/')
def home():
    return "<script> window.location.href = '/flask_files/index.html'</script>"


@app.route('/flask_files/<path:path>')
def send_file(path):
    return send_from_directory('flask_files', path)
  
HTML("<a href='https://{}.{}'>Open Table View</a>".format(os.environ['CDSW_ENGINE_ID'],os.environ['CDSW_DOMAIN']))

if __name__=="__main__":
    app.run(host='127.0.0.1', port=int(os.environ['CDSW_READONLY_PORT']))