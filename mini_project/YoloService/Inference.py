from ultralytics import YOLO
import cv2
import numpy as np
from YoloService import PostProcessing,PreProcessing
from pathlib import Path
import os
import logging


class ImageInference:
    def __init__(self,results_store):
        self.ips = PreProcessing.ImageProcessor()
        self.pps = PostProcessing.PostProcessing()
        self.save_path = Path.cwd() / 'img' / 'result'
        
        self.model_path = '../model/yolo11/Seg/yolo11n-seg.pt'
        self.model = YOLO(self.model_path)
        self.results_store = results_store
        logging.basicConfig(level=logging.INFO)
        
    def predict(self,img_path,job_id):
        img = self.ips.resize(img_path,480)
        
        results = self.model.predict(
            img,
            conf=0.5,
            imgsz=(640,480),
            device='cuda:0',
            max_det=10,
            retina_masks=True,
            )
        
        filename = str(img_path).replace('tmp','result')
        save_file = self.save_path / filename
        
        print('save_file',save_file)
        print('img_path',img_path)
        #이미지 결과 저장
        for i, result in enumerate(results):
            im_plot = result.plot()
            cv2.imwrite(str(save_file), im_plot)
        #이미지 json형태로 전환  
        b64_img = self.pps.image_to_JSON(save_file)
        #전역변수에 값 저장
        self.results_store[job_id] = {
            "message":'추론완료',
            'file': f'{save_file}',
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
            "status": "done"
        }
        logger = logging.getLogger("Inferenc-predict()")
        logger.info('추론완료 및 결과 저장됨')
        # tmp 파일 삭제
        