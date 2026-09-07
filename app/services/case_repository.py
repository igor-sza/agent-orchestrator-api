"""
Carrega os casos criminais que o detetive usa como base factual.

Mora em app/content/ e nao em data/ de proposito. `data/` esta no .gitignore
porque guarda estado de sessao por jogador — efemero e pessoal. Caso criminal e
CONTEUDO do jogo: precisa ser versionado e chegar igual na maquina de todo mundo.
Enquanto o antigo fatos_crime.json era procurado em data/, ele nunca existiu em
lugar nenhum, e o prompt saia dizendo 'Vitima: Desconhecida'.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).resolve().parent.parent / "content" / "cases"

# Usado quando o request nao informa case_id — hoje o Unity nao envia, entao
# este e o caso que roda. Mantem o jogo funcionando sem mudanca no cliente.
CASO_PADRAO = "CASO 01"


def _normalizar(case_id: str) -> str:
    """'CASO 01', 'caso_01' e 'caso-01' devem achar o mesmo arquivo."""
    return "".join(ch for ch in case_id.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _carregar_todos() -> Dict[str, Dict[str, Any]]:
    """
    Le a pasta de casos uma vez. Cacheado porque sao arquivos estaticos lidos a
    cada turno de todo jogador — reler do disco toda vez seria desperdicio.
    """
    casos: Dict[str, Dict[str, Any]] = {}

    if not CASES_DIR.is_dir():
        logger.error(f"Pasta de casos nao encontrada: {CASES_DIR}")
        return casos

    for arquivo in sorted(CASES_DIR.glob("*.json")):
        try:
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Caso invalido em {arquivo.name}: {exc}")
            continue

        case_id = dados.get("case_id")
        if not case_id:
            logger.error(f"{arquivo.name} nao tem 'case_id'; ignorado.")
            continue

        casos[_normalizar(case_id)] = dados
        logger.info(f"Caso carregado: {case_id} — {dados.get('titulo', 'sem titulo')}")

    if not casos:
        logger.error(f"Nenhum caso valido em {CASES_DIR}.")

    return casos


def get_case(case_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Devolve o caso pedido, ou o padrao quando case_id vier vazio.

    Devolve None se nao houver caso nenhum — quem chama decide o que fazer.
    O orquestrador degrada para um prompt sem fatos em vez de derrubar a
    partida, que e o comportamento que ja existia.
    """
    casos = _carregar_todos()
    if not casos:
        return None

    alvo = _normalizar(case_id) if case_id else _normalizar(CASO_PADRAO)

    caso = casos.get(alvo)
    if caso is not None:
        return caso

    if case_id:
        logger.warning(f"case_id '{case_id}' desconhecido; usando o caso padrao.")

    padrao = casos.get(_normalizar(CASO_PADRAO))
    if padrao is not None:
        return padrao

    # Sem o padrao, qualquer caso e melhor que nenhum contexto factual.
    return next(iter(casos.values()))


def list_cases() -> List[Dict[str, str]]:
    """
    Resumo publico dos casos, para o cliente montar a tela de selecao.

    Devolve so o que pode ser mostrado antes do interrogatorio: nada de
    `verdade`, que vazaria o culpado para quem abrisse o endpoint.
    """
    return [
        {
            "case_id": c.get("case_id", ""),
            "titulo": c.get("titulo", ""),
            "resumo": c.get("resumo_publico", ""),
        }
        for c in _carregar_todos().values()
    ]
