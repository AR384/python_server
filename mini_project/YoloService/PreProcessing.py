import cv2
import PIL
import numpy as np
import logging
import imutils

from pathlib import Path

class ImageProcessor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
    
    def __recolor(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #이미지 저장
    def tmp_ImageSave(self,img_byte,job_id):
        logger = logging.getLogger('[ ImageProcessor-tmp_ImageSave ]')
        # uuid.uuid4()는 랜덤하고 유일한 UUID를 생성 jobid와 일치하는 이미지 받기
        tmp_filename = f'tmp_{job_id}.jpg'
        # 현재 실행 중인 프로젝트 루트 경로에 "tmp"라는 폴더
        folder_path = Path.cwd() / 'img' / 'tmp'
        #/img/tmp/tmp_2d018f2f-9e2b-48e0-a6ec-d5ec4c6bbd3f.jpg 처럼 생성
        folder_path.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
        temp_path = folder_path / tmp_filename
        
        img_byte.file.seek(0)
        img_data = img_byte.file.read()
        size = len(img_data)
        
        logger.info(f"이미지 이름: {temp_path}")
        logger.info(f"이미지 크기: {size} bytes")
        
        # "wb"는 바이너리 쓰기 모드 image.file: FastAPI의 UploadFile 객체가 가지고 있는 원본 파일 스트림을 복사
        with open(temp_path,"wb") as f:
            f.write(img_data)
            logger.info("사진 저장 완료")
        #파일 경로 되돌려주기
        return temp_path 
    
    def resize(self, img_path,w):
        logger = logging.getLogger('[ ImageProcessor-resize ]')
        logger.info('이미지 리사이징')
        img = cv2.imread(str(img_path))
        # 바이트를 읽어서 nparr로 만듬->다시 디코딩으로 이미지로 읽음 현재 안씀
        # nparr = np.frombuffer(img,np.uint8)
        # img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        resized_img = imutils.resize(img, height=w)
        return resized_img