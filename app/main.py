import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import InterrogateRequest, ResponseContract
from app.services.context_memory import ContextMemoryManager
from app.services.prompt_orchestrator import PromptOrchestrator

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "time": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(data, ensure_ascii=False)

pasta_logs = Path("logs")
pasta_logs.mkdir(parents=True, exist_ok=True)

arquivo_log = pasta_logs / f'api_{datetime.now().strftime("%Y%m%d")}.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
arquivo = logging.FileHandler(str(arquivo_log), encoding='utf-8')

formatter = JsonFormatter()
console.setFormatter(formatter)
arquivo.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(arquivo)

app = FastAPI(
    title="Guilty API",
    description="API do jogo",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

memoria = ContextMemoryManager(storage_dir="data")
orchestrator = PromptOrchestrator(memory=memoria)

# controle de spam
rate_limit: Dict[str, List[float]] = {}
MAX_REQ = 30
JANELA_TEMPO = 60

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/interrogate", response_model=ResponseContract)
async def interrogate(req: InterrogateRequest):
    sessao = req.session_id
    agora = time.time()

    historico = rate_limit.setdefault(sessao, [])
    historico = [t for t in historico if agora - t <= JANELA_TEMPO]
    rate_limit[sessao] = historico

    if len(historico) >= MAX_REQ:
        logger.warning(f"Sessao {sessao} bloqueada (rate limit)")
        raise HTTPException(status_code=429, detail="Muitas requisicoes")

    historico.append(agora)
    rate_limit[sessao] = historico

    logger.info(f"Mensagem recebida na sessao {sessao}")
    memoria.append_turn(sessao, req.player_text)

    modo = "real" if orchestrator.gemini_client.api_key else "mock"
    logger.info(f"Modo: {modo}")

    res = orchestrator.analyze(sessao, req.player_text, mode=modo)

    memoria.append_turn(sessao, res.get("texto_detetive", ""))

    return res

@app.get("/session/{session_id}")
def get_session(session_id: str):
    return memoria.get_session_info(session_id)

@app.get("/session/{session_id}/history")
def get_history(session_id: str, limit: int = 10):
    content = memoria.load_session_md(session_id)
    linhas = content.splitlines()[-limit:] if content else []
    
    return {
        "session_id": session_id,
        "history": linhas
    }