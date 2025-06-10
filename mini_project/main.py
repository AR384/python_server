from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from YoloService import Inference,PreProcessing

import base64
import uuid
import logging


app = FastAPI()

results = {}
# 결과 저장용 임시 메모리 저장 (실제로는 Redis나 DB를 추천) 
# 전송 next->spring->python  
# python ->db ->spring-> next?
# 결과 next -> spring ->db 
logging.basicConfig(level=logging.INFO)

ips = PreProcessing.ImageProcessor()
inf = Inference.ImageInference(results_store=results)
logger = logging.getLogger("[ APP ]" )

@app.post("/resize") #함수 인자는 뒤에 써야함
async def resize_image(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    #작업아이디 발급
    job_id = str(uuid.uuid4())
    #들어온 사진 임시저장
    img_path = ips.tmp_ImageSave(image,job_id)
    #비동기 백그라운드 테스크 시작
    background_tasks.add_task(inf.predict,img_path,job_id)
    return JSONResponse(content={"message": "FastAPI 이미지 수신 완료", "filename": image.filename,'jobid':job_id})

@app.get("/result/{job_id}")
async def get_result(job_id:str):
    if job_id not in results:
        logger.info("job_id not in results: 데이터를 찾을 수 없음")
        return JSONResponse(status_code=404,content={"message":"FastAPI 작업아이디를 찾을 수 없습니다"})
    return results[job_id]