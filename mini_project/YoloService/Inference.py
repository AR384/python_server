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
    def __init__(self,results_store,jobState):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('[ ImageInference ]')
        self.ips = PreProcessing.ImageProcessor()
        self.pps = PostProcessing.PostProcessing()
        self.model_path = '../model/yolo11/Seg/yolo11x-seg.pt'
        self.model = YOLO(self.model_path)
        self.jobState = jobState
        self.results_store = results_store
        self.inference_result =Results
        self.b64_img = ""
        self.save_file =""
        self.pointlist = []
        self.pred_name = []
        self.viewSize = []
        
    def sequence(self,job_id,tmp_filename):
        try:
            self.logger.info('sequence - 시퀀스 시작')
            self.jobState[job_id] = "processing"
            resized_img,w,h = self.ips.resize(tmp_filename)
            self.viewSize = [w,h]
            diaplay_img_path = self.ips.display_ImageSave(resized_img,tmp_filename)
            self.b64_img = self.pps.image_to_JSON(diaplay_img_path)
            self.inference_result=self.__predict(self.model,resized_img)
            self.__result_IMG_saving(diaplay_img_path,self.inference_result)
            self.pred_name,self.pointlist = self.__result_sorting(job_id,self.results_store,self.inference_result)
            self.__result_push(job_id,self.results_store,self.b64_img,self.pointlist,self.pred_name,self.viewSize)
            self.__reset_state()
            self.jobState[job_id] = "done"
            self.logger.info(f'sequence -Job_id => {job_id}')
            self.logger.info('sequence - 시퀀스 종료')
        except Exception as e:
            self.jobState[job_id] = 'failed'
            self.results_store[job_id] = {"status": "failed", "error": str(e)}
            
    
    def __predict(self,model,resized_img):
        self.logger.info('__predict - 예측 시작')
        results = model.predict(
            resized_img,
            conf=0.5,
            device='cuda:0',
            max_det=10,
            retina_masks=True,
        )
        self.logger.info('__predict - 추론완료')
        return results

    def __result_IMG_saving(self,tmp_filename,inference_result):
        filename = str(tmp_filename).replace('display','result')
        save_path = Path.cwd() / 'img' / 'result'
        save_file = str(save_path / filename) 
        #결과 이미지 저장
        for i, result in enumerate(inference_result):
            im_plot = result.plot()
            cv2.imwrite(save_file, im_plot)
        self.logger.info(f'__result_IMG_saving -결과 이미지 저장 {save_file}')
    
    def __result_sorting(self,job_id,results_store,inference_result):
        pred_name , point_list = [],[]
        #리스트에 작업이 없을 경우 작업
        if job_id not in results_store:
            #전체 레이블 생성
            labels = inference_result[0].names
            for _,result in enumerate(inference_result):
                #인식된 레이블 리스트 플롯
                classified_names = result.boxes.cls.cpu().numpy()
                #익식된 레이블의 폴리곤 만 추출
                mask_coordinate = result.masks.xy
                #레이블에서 값을 받아서 str 리스트로 저장
                for i in classified_names:
                    pred_name.append(labels[int(i)])
                #레이블 별 폴리곤으로 리스트 생성
                for poly in mask_coordinate:
                    # 과부화 걸리면 simple 폴리곤 쓸것
                    # simplified_poly = self.ips.simplify_polygon(poly,tolerance=2.0)
                    # point_str = " ".join(f'{int(x)},{int(y)}' for x,y in simplified_poly) 
                    point_str = " ".join(f'{int(x)},{int(y)}' for x,y in poly)                    
                    point_list.append(point_str)
                
            self.logger.info('__result_sorting - 결과 정리 완료')
        return pred_name , point_list
    
    def __result_push(self,job_id,results_store,b64_img,pointlist,pred_name,viewSize):
        #전역변수에 값 저장
        results_store[job_id] = {
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
            "message":'추론완료',
            "status": "done",
            "poly": pointlist.copy(),
            "names":pred_name.copy(), #리스트 복사해서 새로운 객체로 만듬
            'viewSize':viewSize.copy()
        }
        self.logger.info('__result_push - 데이터 저장 완료')
    
    def __reset_state(self):
        self.pointlist.clear()
        self.pred_name.clear()
        self.viewSize.clear()
        self.logger.info('__reset_state - 내부변수 초기화 완료')