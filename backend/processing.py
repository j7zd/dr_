from readmrz import MrzDetector, MrzReader


def extract_picture_bg_2024(card_front):
    height, width = card_front.shape[:2]
    x1, x2, y1, y2 = 0.02*width, 0.39*width, 0.2*height, 0.96*height
    return card_front[int(y1):int(y2), int(x1):int(x2)]

def read_mrz_bg_2024(card_back):
    detector = MrzDetector()
    reader = MrzReader()

    mrz_region = detector.crop_area(card_back)
    mrz_data = reader.process(mrz_region)
    return mrz_data

