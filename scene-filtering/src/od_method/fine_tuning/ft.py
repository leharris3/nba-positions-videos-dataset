from ultralytics import YOLO

MODEL_PATH = "/mnt/arc/levlevi/nba-positions-videos-dataset/scene-filtering/models/base/yolov8m.pt"
CONFIG_PATH = "/mnt/arc/levlevi/nba-positions-videos-dataset/scene-filtering/_data/yolo_ft_dataset/config.yaml"


def main():
    model = YOLO(MODEL_PATH)
    results = model.train(
        data=CONFIG_PATH, epochs=50, imgsz=640, device=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    # save resulting model
    # all sorts of options available: https://docs.ultralytics.com/modes/export/#how-do-i-enable-int8-quantization-when-exporting-my-yolov8-model
    out_fp = "/mnt/arc/levlevi/nba-positions-videos-dataset/scene-filtering/models/fine-tuned/yolo_sceneparse_m.pt"
    model.save(out_fp)


if __name__ == "__main__":
    main()
