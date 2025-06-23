import cv2
import base64
import logging
import os
import numpy as np
from pathlib import Path
from YoloService import PreProcessing
class PostProcessing:
    def __init__(self,final_result=None):
        logging.basicConfig(level=logging.INFO)
        self.ips = PreProcessing.ImageProcessor()
        self.logger = logging.getLogger("[ PostProcessing ]")
        self.final_result =final_result
        
    def image_to_JSON(self, img_path):
        self.logger.info("image_to_JSON - 이미지 JSON 변환")
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode()
        return b64_img
    
    def temp_Image_Delete(self,):
        root = os.getcwd()
        folder = root+'/img'
        folderlist = os.listdir(folder)
        for i in folderlist:
            if i =='permit':
                continue
            foldername = os.path.join(folder, i)
            filename = os.listdir(foldername)
            for i in filename:
                delete_file = os.path.join(foldername, i)
                if os.path.isfile(delete_file) :
                    os.remove(delete_file)
                    self.logger.info("temp_Image_Delete - 파일을 삭제하였습니다.")
                else:
                    self.logger.error("temp_Image_Delete - 파일을 삭제하지 못 하였습니다.")
    
    def __redraw_mask(self,results,jobid,body):
        self.logger.info("redraw_mask - 마스킹 생성")
        print(body)
        
        path = Path.cwd() / 'img' / 'display'
        imgname = f'display_{jobid}.jpg'
        imgpath = path / imgname
        
        img = cv2.imread(str(imgpath))
        pid = results.get(jobid)
        
        height,width = img.shape[:2]
        
        mask = np.zeros((height, width), dtype=np.uint8)

        for i in range(len(body.selectedIdx)):
            coords_str = pid.get('poly')[body.selectedIdx[i]]
            point = [list(map(int,p.split(','))) for p in coords_str.split()]
            pts = np.array(point,dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        #선택 영역만 색을 부여한 이미지
        masked_image = np.where(mask[:, :, None] == 255, img, gray_3ch)
        #선택 영역 이외 부분 모두 검은색 처리
        bitwise = cv2.bitwise_and(img,img,mask=mask)
        #선택 영역 외곽선
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked_image, contour, -1, (0,255,0), 3)
        #저장 경로 설정
        output_dir = Path.cwd() / 'img' / 'permit'
        output_dir.mkdir(parents=True,exist_ok=True)
        output_path = output_dir / f'permit_{jobid}.jpg'
        
        success = cv2.imwrite(str(output_path), masked_image)
        if success:
            self.logger.info(f"__redraw_mask - 마스크 이미지 저장 완료: {output_path}")
            return output_path
        else:
            self.logger.error(f"__redraw_mask - 마스크 이미지 저장 실패: {output_path}")
    
    def user_selected_img(self,resultDTO,jobid,body):
        self.logger.info("유저 선택 이미지 경로")
        mask_path = self.__redraw_mask(resultDTO,jobid,body)
        base64_img = self.image_to_JSON(mask_path)
        self.final_result[jobid] = {
            'imgPath':base64_img
        }
        self.temp_Image_Delete()
        
    def getImage(self,jobid):
        self.logger.info("image_to_JSON - 이미지 JSON 변환 ")
        root = os.getcwd()
        folder = root+'/img/permit'
        img_path = os.path.join(folder,jobid)
        self.logger.info(f"image_to_JSON - 이미지 JSON 변환 {img_path} ")
        if os.path.isfile(img_path):
            with open(img_path,'rb') as f:
                img_bytes = f.read()
            b64_img = base64.b64encode(img_bytes).decode()
        else:
            b64_img = "이미지 파일이 삭제되 었거나 DB에 존재 하지않습니다."
        return b64_img