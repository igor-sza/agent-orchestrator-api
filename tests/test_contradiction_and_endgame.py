import json
import asyncio
import sys
from pathlib import Path
# Garantir que o diretório do projeto esteja no sys.path para imports locais
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.main import interrogate, memory
from app.schemas import InterrogateRequest
from app.services.context_memory import ContextMemoryManager


def test_contradiction_persistence():
    session = "contradict_session"
    req = InterrogateRequest(session_id=session, player_text="Eu não estava lá.")
    result = asyncio.run(interrogate(req))
    # Verifica arquivo _version.json existe e contém contradições
    version_path = memory._session_version_path(session)
    assert version_path.exists()
    data = json.loads(version_path.read_text(encoding='utf-8'))
    assert 'contradictions' in data
    assert len(data['contradictions']) >= 1
    # O primeiro registro deve referir turno 1
    assert data['contradictions'][-1]['turno'] == 1


def test_endgame_detection_by_history():
    session = "endgame_session"
    # Usa diretamente a memória para simular três níveis altos de suspeita
    mem = ContextMemoryManager(storage_dir="data")
    # Limpar qualquer versão existente para garantir teste limpo
    vp = mem._session_version_path(session)
    try:
        if vp.exists():
            vp.unlink()
    except Exception:
        pass

    mem.append_suspicion(session, 95)
    mem.append_suspicion(session, 92)
    mem.append_suspicion(session, 90)
    recent = mem.get_recent_suspicions(session, n=3)
    assert len(recent) >= 3
    assert all(v >= 90 for v in recent[-3:])
