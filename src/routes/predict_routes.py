from flask import Blueprint
from src.controllers.predict_controller import prediction_controller

predict_bp = Blueprint('predict_bp', __name__, url_prefix='/vinicius_rubens/api')
predict_bp.route('/predict', methods=['POST'])(prediction_controller.predict)