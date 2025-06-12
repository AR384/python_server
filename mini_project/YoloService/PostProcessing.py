import cv2
import base64
import logging
import os
from YoloService import PreProcessing
class PostProcessing:
    def __init__(self,final_result=None):
        self.final_result = final_result
        logging.basicConfig(level=logging.INFO)
        self.ips = PreProcessing.ImageProcessor()
        self.logger = logging.getLogger("[ PostProcessing ]")
        
    def image_to_JSON(self, img_path):
        self.logger.info("이미지 JSON 변환")
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode()
        return b64_img
    
    def image_to_JSON_Original(self, img_path):
        self.logger.info("이미지 JSON 변환")
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode()
        return b64_img
    
    def temp_Image_Delete(self, img_path,foler_path):
        delete_file = foler_path+'/'+img_path
        if os.path.isfile(delete_file):
            os.remove(delete_file)
        else:
            self.logger.error("파일을 삭제하지 못 하였습니다.")
            
    def user_selected_img(self,results,jobid,):
        self.ips.redraw_mask(results,jobid)