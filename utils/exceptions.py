class ChurnPredictionError(Exception): 
    pass 
 
class DataLoadError(ChurnPredictionError): 
    pass 
 
class DataProcessError(ChurnPredictionError): 
    pass 
 
class ModelTrainingError(ChurnPredictionError): 
    pass 
 
class ModelPredictionError(ChurnPredictionError): 
    pass 
 
class ValidationError(ChurnPredictionError): 
    pass 
