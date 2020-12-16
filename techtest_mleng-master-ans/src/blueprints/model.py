import sys
import traceback

import flask
import numpy as np
import json

from flask import Blueprint, current_app, request

'''
Blueprint for single model invocations
'''

bp = Blueprint('model', __name__)

# ------------------------------------------------------------------

'''
Recommender class to take care of data pre-loading and recommendation computation
'''

class Recommender():
    def __init__(self):
        
        self.__items, vectors = [], []
        
        with open('/opt/data/vectors.csv', 'r') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                else:
                    line = line.split(',')
                    self.__items.append(line[0])
                    vectors.append(list(map(float, line[1:])))
        
        self.__data = np.array(vectors)
        
    def query(self, q):
        scores = np.dot(self.__data, q)
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            return {'best': None, 'best_sim': None, 'worst': None, 'worst_sim': None}
        else:
            best = np.argmax(scores)
            worst = np.argmin(scores)
            return {'best': self.__items[best], 'best_sim': scores[best], 'worst': self.__items[worst], 'worst_sim': scores[worst]}
        
    def batch_query(self, qs):
        scores = np.dot(self.__data, qs.T)
        
        nan_flags = np.any(np.isnan(scores), axis=0)
        inf_flags = np.any(np.isinf(scores), axis=0)
        
        bests = np.argmax(scores, 0)
        worsts = np.argmin(scores, 0)
        
        invalid = {'best': None, 'best_sim': None, 'worst': None, 'worst_sim': None}
        
        results = [
            invalid if (nan or inf) else {'best': self.__items[b], 'best_sim': scores[b, i], 'worst': self.__items[w], 'worst_sim': scores[w, i]} \
            for i, (b, w, nan, inf) in enumerate(zip(bests, worsts, nan_flags, inf_flags))
        ]
        
        return results
    
foodProphet = Recommender()

# ------------------------------------------------------------------

@bp.route('/batch', methods=['POST'])
def batch():
    try:
        body = json.loads(request.get_json())
        vectors = np.array([itm['vector'] for itm in body])
    except Exception as e:
        exc_info = sys.exc_info()
        err = ''.join(traceback.format_exception(*exc_info))
        current_app.logger.info("Error during batch request:")
        current_app.logger.info(err)
        return flask.jsonify({'bad request': err}), 400
    batch_result = foodProphet.batch_query(vectors)
    return flask.jsonify(batch_result)
 

@bp.route('/infer', methods=['POST'])
def infer():
    try:
        body = json.loads(request.get_json())
        vector = np.array(body['vector'])
    except Exception as e:
        exc_info = sys.exc_info()
        err = ''.join(traceback.format_exception(*exc_info))
        current_app.logger.info("Error during batch request:")
        current_app.logger.info(err)
        return flask.jsonify({'bad request': err}), 400
    query_result = foodProphet.query(vector)
    return flask.jsonify(query_result)
