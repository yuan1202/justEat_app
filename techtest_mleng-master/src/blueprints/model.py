import flask
import numpy as np
import json

from flask import Blueprint, current_app, request

bp = Blueprint('model', __name__)

'''
Blueprint for single model invocations
'''

file_location = '/opt/data/vectors.csv'

@bp.route('/batch', methods=['POST'])
def batch():
    # TODO -> Create endpoint for batch processing
    pass

@bp.route('/infer', methods=['POST'])
def infer():
    body = json.loads(request.get_json())
    restaurant_similarity = {}

    top_dish_sim = None
    top_dish_name = None
    worst_dish_sim = None
    worst_dish_name = None

    with open(file_location, 'r') as f:
        data = f.readline()
        while data:
            data = f.readline()
            elements = data.split(',')
            name = elements.pop(0)
            if len(elements) == 0:
                break
            restaurant_similarity[name] = np.dot(body['vector'], list(map(float, elements)))
            
        for k, v in restaurant_similarity.items():
            if worst_dish_sim is None or restaurant_similarity[k] <= worst_dish_sim:
                worst_dish_name = k
                worst_dish_sim = v
            if top_dish_sim is None or restaurant_similarity[k] >= top_dish_sim:
                top_dish_name = k
                top_dish_sim = v

    return flask.jsonify({
        "best": top_dish_name,
        "best_sim": top_dish_sim,
        "worst": worst_dish_name,
        "worst_sim": worst_dish_sim
    })
