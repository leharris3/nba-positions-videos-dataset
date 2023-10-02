from ultralytics import YOLO
from paddleocr import PaddleOCR

MODEL_PATH = r"models\yolo\weights\45_ep_lar_full.pt"


class Models():

    def __init__(self) -> None:
        self.yolo = None
        self.paddle_ocr = None
        self.models_loaded = False

    def load(self):
        self.yolo = YOLO(MODEL_PATH)
        self.paddle_ocr = PaddleOCR(use_angle_cls=True,
                                    lang='en',
                                    show_log=False,
                                    det_db_score_mode='slow',
                                    ocr_version='PP-OCRv4',
                                    rec_algorithm='SVTR_LCNet',
                                    drop_score=0.0,
                                    )
        self.models_loaded = True
