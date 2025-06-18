from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, BackgroundTasks,Form
from YoloService import Inference,PreProcessing,PostProcessing,StateHandler
from pydantic import BaseModel
from MyDTO import CustomDTO
import uuid
import logging

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

@app.post("/fastapi/inference") #함수 인자는 뒤에 써야함
async def inference_image(background_tasks: BackgroundTasks, image: UploadFile = File(...),username:str = Form(...)):
    #작업아이디 발급
    jobid = str(uuid.uuid4())

    jobState[jobid] = "queued"
    #들어온 사진 임시저장 [원본]
    tmp_filename = await ips.tmp_ImageSave(image,jobid)
    response_data = CustomDTO.ImageUploadResponseDTO(jobid=jobid,username=username)
    #비동기 백그라운드 테스크 시작
    background_tasks.add_task(inf.sequence,jobid,tmp_filename)
    return CustomDTO.ApiResponseDTO(status=jobState.get(jobid),message="Fast API 이미지 수신 완료",data=response_data)

@app.get('/fastapi/inference/{jobid}/result')
async def get_result(jobid:str):
    return await sth.handle(jobid)

@app.post('/fastapi/inference/{jobid}/permission')
async def users_selected(jobid:str, body:CustomDTO.ImagePermitRequestDTO ,background_tasks:BackgroundTasks):
    logger.info("사용자 승인 이미지 처리 요청 body =>",body)
    print(body.model_dump())
    response_data = CustomDTO.ImagePermitResponseDTO(jobid=jobid,selectedIdx=body.selectedIdx,selectedname=body.selectedname)
    if jobid not in resultDTO:
        logger.info("@app.post('/fastapi/inference/jobid/permission') jobid not in results: 데이터를 찾을 수 없음")
        return CustomDTO.ApiResponseDTO(status='failed',message='해당하는 jobid에 해당하는 결과가 없습니다.',data=None)
    background_tasks.add_task(pps.user_selected_img,resultDTO,jobid,body)
    return CustomDTO.ApiResponseDTO(status="que",message='Fast API 데이터 수신 완료', data=response_data)

@app.get('/fastapi/inference/{jobid}/permission/result')
async def final_sending(jobid:str):
    logger.info("사용자 승인 이미지 처리 결과 반환")
    if jobid not in resultDTO:
        logger.info("@app.get('/final-result/jobid') - jobid not in results: 데이터를 찾을 수 없음")
        return CustomDTO.ApiResponseDTO(status="failed",message='해당하는 jobid에 해당하는 결과가 없습니다.',data=None)
    response_data = CustomDTO.ImagePermitResultDTO()
    logger.info(f'jobid : {jobid} 작업 종료  반환 후 메모리에서 삭제')
    return None