PATH_TO_TESSERACT = r"/usr/local/bin/pytesseract"
PRINT_FRAME_OFFSET = 1000
LARGE_STEP = 25
MOD_STEP = 5
BREAK_POINT = -1
QUARTER_CONFIG = r'--oem 1 --psm 10 -c tessedit_char_whitelist=1234 load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0'
CLOCK_CONFIG = r'--oem 1 --psm 3 -c tessedit_char_whitelist=0123456789.: -c load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0'
TRIM_VIDEO_BITRATE = 1000000

ROIS = {
    "TNT_QUARTER": {"x_start": 866, "width": 14, "y_start": 1168-560, "height": 24},
    "TNT_CLOCK": {"x_start": 954, "width": 63, "y_start": 1168-560, "height": 24},
    "ESP_QUARTER": {"x_start": 835, "width": 14, "y_start": 1138-560, "height": 24},
    "ESP_CLOCK": {"x_start": 885, "width": 62, "y_start": 1138-560, "height": 24},
    "FOX_QUARTER": {"x_start": 840, "width": 10, "y_start": 1158-560, "height": 28},
    "FOX_CLOCK": {"x_start": 946, "width": 60, "y_start": 1158-560, "height": 28},
    "CSN_QUARTER": {"x_start": 1005, "width": 129, "y_start": 1183-560, "height": 28},
    "CSN_CLOCK": {"x_start": 1005, "width": 129, "y_start": 1183-560, "height": 28},
    "TSN_QUARTER": {"x_start": 978, "width": 157, "y_start": 1180-560, "height": 23},
    "TSN_CLOCK": {"x_start": 978, "width": 157, "y_start": 1180-560, "height": 23}
}
