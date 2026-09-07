"""
Parsing da resposta textual do LLM para o dicionario do contrato.

Vive fora dos provedores de proposito: transformar texto em JSON nao tem nada
de especifico do Gemini. O Ollama devolve exatamente as mesmas patologias
(cerca ```json em volta, texto solto quando o modelo ignora a instrucao), entao
manter isso aqui evita que cada provedor carregue a sua copia divergente.
"""
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _fallback_contract(response_text: str) -> Dict[str, Any]:
    """
    Ultimo recurso: o modelo respondeu, mas nao em JSON. Em vez de estourar,
    devolve o texto dele como fala do detetive num contrato valido — o jogo
    continua rodando e o jogador ve a resposta, ainda que sem os metadados.
    """
    return {
        "id_turno": 1,
        "texto_detetive": response_text.strip(),
        "status_investigacao": {
            "nivel_suspeita": 50,
            "congelar_input": False,
            "detectou_mentira": "mentira" in response_text.lower(),
            "fim_de_jogo": False,
        },
        "feedback_visual": {
            "cor_iluminacao": "#FFFF00",
            "bpm_musica": 100,
            "animacao_trigger": "Thinking",
        },
    }


def _strip_code_fence(text: str) -> str:
    """
    Remove a cerca de codigo em volta do JSON.

    Aceita ```json e ``` puro: o Gemini costuma usar o primeiro e o Ollama o
    segundo, e a versao anterior so tratava ```json — com o Ollama isso cairia
    direto no fallback em toda resposta cercada.
    """
    limpo = text.strip()
    if not limpo.startswith("```"):
        return limpo

    primeira_quebra = limpo.find("\n")
    if primeira_quebra == -1:
        return limpo

    corpo = limpo[primeira_quebra + 1:]
    fim = corpo.rfind("```")
    return corpo[:fim].strip() if fim != -1 else corpo.strip()


def parse_llm_json(response_text: str) -> Dict[str, Any]:
    """
    Converte a saida do modelo no dicionario do contrato.

    Nunca levanta excecao: se nada for aproveitavel, devolve o contrato de
    fallback. Quem chama decide se o conteudo serve, validando com o
    ResponseContract.
    """
    if not response_text or not response_text.strip():
        return _fallback_contract("")

    for candidato in (response_text, _strip_code_fence(response_text)):
        try:
            dados = json.loads(candidato)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(dados, dict):
            return dados

    logger.warning("Resposta do modelo nao era JSON valido; usando fallback.")
    return _fallback_contract(response_text)
