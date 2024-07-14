from typing import Dict
from ultralytics import YOLO

YOLO_MODEL_PATH = "/playpen-storage/levlevi/nba-positions-videos-dataset/statvu_alignment/models/yolo/weights/tr_roi_finetune_60_large.pt"

class YOLOModel:
    
    _model = None
        
    @classmethod
    def get_model(cls, config: Dict):
        device = config["device"]
        verbose = config["yolo"]["verbose"]
        if cls._model is None:
            cls._model = YOLO(YOLO_MODEL_PATH, verbose=verbose)
        return cls._model.to(device)
        
