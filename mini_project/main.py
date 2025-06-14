from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from YoloService import Inference,PreProcessing,PostProcessing,StateHandler
from pydantic import BaseModel
import uuid
import logging


class Recieve(BaseModel):
    selectedIdx : list
    selectedname : list
    jobid: str

app = FastAPI()

resultDTO = {}
jobState = {}



# 결과 저장용 임시 메모리 저장 (실제로는 Redis나 DB를 추천) 
# 전송 next->spring->python  
# python ->db ->spring-> next?
# 결과 next -> spring ->db 
logging.basicConfig(level=logging.INFO)

ips = PreProcessing.ImageProcessor()
inf = Inference.ImageInference(results_store=resultDTO,jobState=jobState)
pps = PostProcessing.PostProcessing()
sth = StateHandler.StateHandler(resultDTO,jobState)
logger = logging.getLogger("[ APP ]" )

@app.post("/inference") #함수 인자는 뒤에 써야함
async def inference_image(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    #작업아이디 발급
    job_id = str(uuid.uuid4())
    jobState[job_id] = "queued"
    #들어온 사진 임시저장 [원본]
    tmp_filename = await ips.tmp_ImageSave(image,job_id)
    #비동기 백그라운드 테스크 시작
    background_tasks.add_task(inf.sequence,job_id,tmp_filename)
    return JSONResponse(content={"message": "[ FastAPI ]  이미지 수신 완료", "filename": image.filename,'jobid':job_id})

@app.get('/result/{job_id}')
async def get_result(job_id:str):
    return await sth.handle(job_id)

@app.post('/userselected/img/{job_id}')
async def users_selected(job_id:str, body:Recieve ,background_tasks:BackgroundTasks):
    logger.info("사용자 승인 이미지 처리 요청")
    body.model_dump()
    if job_id not in resultDTO:
        logger.info("@app.post('/userselected/img/job_id') job_id not in results: 데이터를 찾을 수 없음")
    background_tasks.add_task(pps.user_selected_img,resultDTO,job_id,body)
    return JSONResponse(content={"message": "[ FastAPI ]  데이터 수신 완료", "requestBody": body.model_dump() ,'jobid':job_id})

@app.get('/final-result/{job_id}')
async def final_sending(job_id:str):
    logger.info("사용자 승인 이미지 처리 결과 반환")
    if job_id not in resultDTO:
        logger.info("@app.get('/final-result/job_id') - job_id not in results: 데이터를 찾을 수 없음")
        return JSONResponse(status_code=404,content={"message":"[ FastAPI ] 작업아이디를 찾을 수 없습니다"})
    result_data = resultDTO.pop(job_id)
    logger.info(f'job_id : {job_id} 작업 종료  반환 후 메모리에서 삭제')
    return result_data