import os
import flask
import logging
import sys

from flask import request, jsonify, abort

# Logging
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])

# The flask app for serving predictions
app = flask.Flask(__name__)

# Healthchecks
from blueprints import healthcheck
app.logger.info("Adding healthcheck blueprint")
app.register_blueprint(healthcheck.bp)

# Model load
app.logger.info("Adding model blueprint")
from blueprints import model
app.register_blueprint(model.bp)
