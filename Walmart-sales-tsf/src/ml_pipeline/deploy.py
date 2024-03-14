from flask import Flask
from flask_restful import Resource, Api
from . import Utils


# Create flask application

app = Flask(__name__)
api = Api(app)

# Loanpred class for handling api req

class LoanPred(Resource):
    def __init__(self,model):
        self.model = model

    def get(self):
        return {"ans" : "sucess"}

def init(load_path):
    uploaded_model = Utils.load_model(load_path)
    api.add_resource(LoanPred, "/", resource_class_kwargs = {"model": uploaded_model})
    app.run(port=12345)

