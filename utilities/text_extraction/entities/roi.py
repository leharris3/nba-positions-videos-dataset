class ROI:

    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float) -> None:
        """4 coordinate boudning box assoicated with an ROI."""

        assert x1 > 0 and y1 > 0, "Cords must be greater than 0."
        assert x1 < x2, "X1 must be less than X2."
        assert y1 < y2, "Y1 must be less than Y2."
        print(y1, y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.confidence = confidence
        self.width = x2 - x1
        self.height = y2 - y1
