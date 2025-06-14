from fastapi.responses import JSONResponse

class StateHandler:
    def __init__(self, result_store, state_store):
        self.resultDTO = result_store
        self.jobState = state_store

        # 상태별 처리 메서드 매핑
        self.state_handlers = {
            None: self.handle_not_found,
            "not_found": self.handle_not_found,
            "queued": self.handle_processing,
            "processing": self.handle_processing,
            "final_processing": self.handle_processing,
            "failed": self.handle_failed,
            "done": self.handle_done,
            "final_done": self.handle_done,
        }

    async def handle_not_found(self, job_id):
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "message": "작업 아이디를 찾을 수 없습니다."}
        )

    async def handle_processing(self, job_id):
        return JSONResponse(
            status_code=202,
            content={"status": "processing", "message": "작업이 진행 중입니다."}
        )

    async def handle_failed(self, job_id):
        content = self.resultDTO.get(job_id, {"status": "failed", "message": "작업 실패"})
        return JSONResponse(status_code=500, content=content)

    async def handle_done(self, job_id):
        data = self.resultDTO.get(job_id)
        if data is None:
            return await self.handle_not_found(job_id)
        return JSONResponse(status_code=200, content=data)

    async def handle(self, job_id):
        state = self.jobState.get(job_id)
        handler = self.state_handlers.get(state)

        if handler is None:
            return JSONResponse(
                status_code=500,
                content={"status": "unknown", "message": "정의되지 않은 상태입니다."}
            )
        return await handler(job_id)