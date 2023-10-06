from ultralytics import YOLO
from paddleocr import PaddleOCR

MODEL_PATH = r"C:\Users\Levi\Desktop\__dep__\contextualized-shot-quality-estimation\models\yolo\weights\tr_roi_finetune_130_large.pt"


class Models:

    yolo = yolo = YOLO(MODEL_PATH)
    paddle_ocr = PaddleOCR(use_angle_cls=True,
                           lang='en',
                           show_log=False,
                           det_db_score_mode='slow',
                           ocr_version='PP-OCRv4',
                           rec_algorithm='SVTR_LCNet',
                           drop_score=0.8,
                           )
