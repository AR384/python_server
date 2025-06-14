import cv2
import PIL
import numpy as np
import logging
import imutils
from shapely.geometry import Polygon,MultiPolygon
from pathlib import Path

class ImageProcessor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('[ ImageProcessor ]')
        
    def __recolor(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #이미지 저장
    async def tmp_ImageSave(self,img_byte,job_id):
        self.logger.info('tmp_ImageSave - 임시파일 저장')
        # uuid.uuid4()는 랜덤하고 유일한 UUID를 생성 jobid와 일치하는 이미지 받기
        tmp_filename = f'tmp_{job_id}.jpg'
        # 현재 실행 중인 프로젝트 루트 경로에 "tmp"라는 폴더
        folder_path = Path.cwd() / 'img' / 'tmp'
        #/img/tmp/tmp_2d018f2f-9e2b-48e0-a6ec-d5ec4c6bbd3f.jpg 처럼 생성
        folder_path.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
        temp_path = folder_path / tmp_filename
        
        img_byte.file.seek(0)
        # img_data = img_byte.file.read()
        img_data = img_byte.file.read()
        size = len(img_data)
        
        self.logger.info(f"tmp_ImageSave - 이미지 이름: {temp_path}")
        self.logger.info(f"tmp_ImageSave - 이미지 크기: {size} bytes")
        
        # "wb"는 바이너리 쓰기 모드 image.file: FastAPI의 UploadFile 객체가 가지고 있는 원본 파일 스트림을 복사
        with open(temp_path,"wb") as f:
            f.write(img_data)
            
        #파일 경로 되돌려주기
        self.logger.info(f"tmp_ImageSave - 사진 저장 완료 - {temp_path}")
        return temp_path
    
    def display_ImageSave(self,img,tmp_filename):
        
        display_filename = str(tmp_filename).replace('tmp','display')
        
        folder_path = Path.cwd() / 'img' / 'display'
        folder_path.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
        temp_path = folder_path / display_filename
        
        cv2.imwrite(temp_path, img)
        self.logger.info(f"display_ImageSave - 디스플레이 사진 저장 완료 {temp_path}")
        return temp_path
    
    def resize(self, img_path,one_side_size=640):
        self.logger.info(f'resize 이미지 리사이징 {img_path}')
        img = cv2.imread(str(img_path))
        hr, wr = img.shape[:2]
        if hr>wr : #세로 사진 기준으로 처리
            new_height = one_side_size
            ratio = new_height / hr
            new_width = int(wr * ratio)
        else: # 가로 사진으로 처리
            new_width = one_side_size
            ratio = new_width / wr
            new_height = int(hr * ratio)
            
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        self.logger.info(f'resize -  이미지 리사이징 ={new_width,new_height}')
        return resized_img , new_width , new_height
    
    def simplify_polygon(self,coords, tolerance=2.0):
        self.logger.info('simplify_polygon - 폴리곤 처리 ')
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # Invalid geometry fix
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        
        # 결과가 Polygon이면 그대로 사용
        if isinstance(simplified, Polygon):
            return list(simplified.exterior.coords)
        # 결과가 MultiPolygon이면 가장 큰 polygon 하나만 사용
        elif isinstance(simplified, MultiPolygon):
            largest = max(simplified.geoms, key=lambda p: p.area)
            return list(largest.exterior.coords)
        
        else:
            raise ValueError("Simplified geometry is neither Polygon nor MultiPolygon.")
        
    def redraw_mask(self,results,jobid,body):
        self.logger.info("redraw_mask - 마스킹 생성")
        print("받은 타입" ,type(body))
        print(body.selectedname)
        print(body)
        pid = results.get(jobid)
        print('특정번호 폴리곤',pid.get('poly')[body.selectedIdx[0]])
        print('전체 폴리',pid.get('poly'))
        print('이름 : ',pid.get('names'))
