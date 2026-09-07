import json
import asyncio
import sys
from pathlib import Path
# Garantir que o diretório do projeto esteja no sys.path para imports locais
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.main import interrogate
from app.schemas import InterrogateRequest
from app.services.llm_providers import MockProvider


def test_interrogate_endpoint_mock():
    req = InterrogateRequest(session_id="test_session", player_text="Eu não estava na biblioteca.")
    # Chamando a função direto (sem passar pelo FastAPI), o Depends não é
    # resolvido — o provedor precisa ser passado à mão.
    result = asyncio.run(interrogate(req, provider=MockProvider()))
    # Deve retornar dicionário compatível com ResponseContract
    assert isinstance(result, dict)
    assert "id_turno" in result
    assert "texto_detetive" in result
    assert "status_investigacao" in result
    assert "feedback_visual" in result
    # Campos de status
    status = result["status_investigacao"]
    assert isinstance(status.get("nivel_suspeita"), int)
    assert isinstance(status.get("detectou_mentira"), bool)
    # Validar com Pydantic
    from app.schemas import ResponseContract
    ResponseContract(**result)
