from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import numpy as np
from YoloService import PostProcessing,PreProcessing
from pathlib import Path
import os
import logging
from shapely.geometry import Polygon

class ImageInference:
    def __init__(self,results_store):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ImageInference')
        self.ips = PreProcessing.ImageProcessor()
        self.pps = PostProcessing.PostProcessing()
        self.save_path = Path.cwd() / 'img' / 'result'
        self.model_path = '../model/yolo11/Seg/yolo11x-seg.pt'
        self.model = YOLO(self.model_path)
        self.results_store = results_store
        self.inference_result =Results
        self.b64_img = ""
        self.save_file =""
        self.pointlist = []
        self.pred_name = []
        
    
    def Sequence(self,img_path,job_id):
        self.logger.info('시퀀스 시작')
        self.__predict(img_path,job_id)
        self.__result_IMG_saving_coverting(img_path)
        self.__result_sorting()
        self.__result_push(job_id,)
        self.logger.info('시퀀스 종료')
    
    def __predict(self,img_path,job_id):
        img = self.ips.resize(img_path,480)
        results = self.model.predict(
            img,
            conf=0.5,
            imgsz=(640,480),
            device='cuda:0',
            max_det=10,
            retina_masks=True,
        )
        self.inference_result=results
        self.logger.info('추론완료')

    def __result_IMG_saving_coverting(self,img_path):
        filename = str(img_path).replace('tmp','result')
        self.save_file = str(self.save_path / filename) 
        print('save_file : ',self.save_file)
        print('img_path : ',img_path)
        #이미지 결과 저장
        for i, result in enumerate(self.inference_result):
            im_plot = result.plot()
            cv2.imwrite(self.save_file, im_plot)
        #이미지 json형태로 전환  
        self.b64_img = self.pps.image_to_JSON(self.save_file)
        self.logger.info('이미지 저장 및 json인코딩 완료')
    
    def __result_sorting(self):
        #전체 레이블 생성
        labels = self.inference_result[0].names
        for _,result in enumerate(self.inference_result):
            #인식된 레이블 리스트 플롯
            classified_names = result.boxes.cls.cpu().numpy()
            #익식된 레이블의 폴리곤 만 추출
            mask_coordinate = result.masks.xy
            #레이블에서 값을 받아서 str 리스트로 저장
            for i in classified_names:
                self.pred_name.append(labels[int(i)])
            #레이블 별 폴리곤으로 리스트 생성
            for poly in mask_coordinate:
                simplified_poly = self.ips.simplify_polygon(poly,tolerance=2.0)
                point_str = " ".join(f'{int(x)},{int(y)}' for x,y in simplified_poly)
                self.pointlist.append(point_str)
            
        self.logger.info('결과 정리 완료')
    
    def __result_push(self,job_id):
        #전역변수에 값 저장
        self.results_store[job_id] = {
            "message":'추론완료',
            'file': f'{self.save_file}',
            "image_base64": f"data:image/jpeg;base64,{self.b64_img}",
            "status": "done",
            "poly": self.pointlist,
            "names":self.pred_name
        }
        self.logger.info('데이터 저장 완료')
    
    