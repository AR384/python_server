from fastapi import HTTPException
from MyDTO.CustomDTO import ApiResponseDTO, ImageProcessResultDTO  # 실제 경로에 맞게 조정
import logging

class StateHandler:
    def __init__(self, result_store, state_store):
        self.logger = logging.getLogger("[ StateHandler ]")
        self.resultDTO = result_store
        self.jobState = state_store
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
        return ApiResponseDTO[None](
            status="not_found",
            message="작업 아이디를 찾을 수 없습니다.",
            data=None
        )

    async def handle_processing(self, job_id):
        return ApiResponseDTO[None](
            status="processing",
            message="작업이 진행 중입니다.",
            data=None
        )

    async def handle_failed(self, job_id):
        raw = self.resultDTO.get(job_id, {})
        return ApiResponseDTO[None](
            status=raw.get("status", "failed"),
            message=raw.get("message", "작업 실패"),
            data=None
        )

    async def handle_done(self, job_id):
        raw = self.resultDTO.get(job_id)
        if raw is None:
            return await self.handle_not_found(job_id)

        # resultDTO는 dict이므로 ImageProcessResultDTO로 변환
        result_data = ImageProcessResultDTO(**raw)

        return ApiResponseDTO[ImageProcessResultDTO](
            status="done",
            message="작업 완료",
            data=result_data
        )

    async def handle(self, job_id):
        state = self.jobState.get(job_id)
        handler = self.state_handlers.get(state)
        self.logger.info(f'handle - 작업 상태 = {state}')
        if handler is None:
            return ApiResponseDTO[None](
                status="unknown",
                message="정의되지 않은 상태입니다.",
                data=None
            )
        return await handler(job_id)
