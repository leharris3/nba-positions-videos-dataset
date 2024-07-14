import torch
from ultralytics import YOLO

YOLO_MODEL_PATH = r"/mnt/opr/levlevi/nba-positions-videos-dataset/models/yolo/weights/tr_roi_finetune_60_large.pt"
TR_OCR_MODEL_PATH = "microsoft/trocr-base-stage1"

class YOLOModel:
    
    global _model
    _model = None
        
    @staticmethod
    def get_model(device: int = 0):
        if YOLOModel._model == None:
            YOLOModel._model = YOLO(YOLO_MODEL_PATH)
        return YOLOModel._model.to(device)
        
