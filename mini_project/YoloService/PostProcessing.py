import cv2
import base64
import logging
import os
from YoloService import PreProcessing
class PostProcessing:
    def __init__(self,final_result=None):
        logging.basicConfig(level=logging.INFO)
        self.ips = PreProcessing.ImageProcessor()
        self.logger = logging.getLogger("[ PostProcessing ]")
        
    def image_to_JSON(self, img_path):
        self.logger.info("image_to_JSON - 이미지 JSON 변환")
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode()
        return b64_img
    
    def temp_Image_Delete(self, img_path,foler_path):
        self.logger.error("temp_Image_Delete - 파일을 삭제하였습니다.")
        delete_file = foler_path+'/'+img_path
        if os.path.isfile(delete_file):
            os.remove(delete_file)
        else:
            self.logger.error("temp_Image_Delete - 파일을 삭제하지 못 하였습니다.")
            
    def user_selected_img(self,resultDTO,jobid,body):
        mask_path = self.ips.redraw_mask(resultDTO,jobid,body)
        base64_img = self.image_to_JSON(mask_path)
        