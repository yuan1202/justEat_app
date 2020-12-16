import flask

from flask import Blueprint

bp = Blueprint('healthcheck', __name__)

@bp.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy.
    """

    return flask.Response(status=200, mimetype='application/json')