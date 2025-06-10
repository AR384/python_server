import cv2
import base64
import logging
import os

class PostProcessing:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
    
    def image_to_JSON(self, img_path):
        logger = logging.getLogger("[ PostProcessing-image_to_JSON ]")
        logger.info("이미지 JSON 변환")
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode()
        return b64_img
    
    def temp_Image_Delete(self, img_path,foler_path):
        delete_file = foler_path+'/'+img_path
        if os.path.isfile(delete_file):
            os.remove(delete_file)
        else:
            logger = logging.getLogger("[ PostProcessing-temp_Image_Delete ]")
            logger.error("파일을 삭제하지 못 하였습니다.")