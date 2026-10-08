import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import InterrogateRequest, ResponseContract
from app.services.case_repository import list_cases
from app.services.context_memory import ContextMemoryManager
from app.services.llm_providers import (
    LLMProvider,
    LLMUnavailableError,
    get_llm_provider,
)
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

# Handlers no logger do pacote "app", nao no deste modulo: pendurados so no
# app.main, os logs de app.services.* (resposta bruta do modelo, tempo e tok/s
# do llama.cpp) eram descartados — nao chegavam nem ao console nem ao arquivo.
logger_app = logging.getLogger("app")
logger_app.setLevel(logging.INFO)

console = logging.StreamHandler()
arquivo = logging.FileHandler(str(arquivo_log), encoding='utf-8')

formatter = JsonFormatter()
console.setFormatter(formatter)
arquivo.setFormatter(formatter)

logger_app.addHandler(console)
logger_app.addHandler(arquivo)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # O provedor local mantem um httpx.AsyncClient aberto entre requisicoes;
    # sem fechar no shutdown o uvicorn reclama de conexao vazando no reload.
    await get_llm_provider().aclose()

app = FastAPI(
    title="Guilty API",
    description="API do jogo",
    version="1.0",
    lifespan=lifespan
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
async def interrogate(req: InterrogateRequest,
                      provider: LLMProvider = Depends(get_llm_provider)):
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
    memoria.append_turn(sessao, req.player_text, role="suspeito")

    logger.info(f"Provedor: {provider.nome}")

    try:
        res = await orchestrator.analyze(sessao, req.player_text,
                                         provider=provider, case_id=req.case_id)
    except LLMUnavailableError as exc:
        # 503 e nao 500: o problema nao e a nossa API, e o modelo que nao esta
        # no ar. O ApiClient do Unity ja trata nao-2xx e mostra a mensagem.
        logger.error(f"Provedor {provider.nome} indisponivel: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    memoria.append_turn(sessao, res.get("texto_detetive", ""), role="detetive")

    return res

@app.get("/cases")
def listar_casos():
    """
    Casos disponiveis, so com o que pode ser mostrado antes do interrogatorio.
    A secao `verdade` nunca sai daqui — vazaria o culpado.
    """
    return {"cases": list_cases()}

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