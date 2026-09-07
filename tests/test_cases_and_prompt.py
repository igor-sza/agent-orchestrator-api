"""
Cobre a correcao do problema que deixava o detetive sem base factual:
o prompt precisa carregar os fatos do caso, e a memoria precisa dizer
quem falou o que.
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.case_repository import get_case, list_cases
from app.services.context_memory import ContextMemoryManager
from app.services.prompt_orchestrator import PromptOrchestrator


def _orchestrator():
    return PromptOrchestrator(memory=ContextMemoryManager(storage_dir="data"))


def test_existe_caso_padrao():
    caso = get_case()
    assert caso is not None, "nenhum caso carregado — a pasta app/content/cases esta vazia?"
    assert caso["case_id"] == "CASO 01"
    assert caso["verdade"]["culpado"]
    assert caso["evidencias_conhecidas"]


def test_case_id_desconhecido_cai_no_padrao():
    # O Unity ainda nao envia case_id; um id errado nao pode derrubar a partida.
    assert get_case("CASO INEXISTENTE")["case_id"] == "CASO 01"
    # E a normalizacao aceita as variacoes de escrita do mesmo id.
    assert get_case("caso_01")["case_id"] == "CASO 01"
    assert get_case("caso-01")["case_id"] == "CASO 01"


def test_listagem_publica_nao_vaza_a_verdade():
    for resumo in list_cases():
        assert "verdade" not in resumo
        assert set(resumo) == {"case_id", "titulo", "resumo"}


def test_prompt_carrega_os_fatos_do_caso():
    caso = get_case()
    prompt = _orchestrator().build_prompt("sess_prompt", "Eu sai as 19h.", caso)

    # O bug original: estes tres literais apareciam no lugar dos fatos.
    assert "Vítima: Desconhecida" not in prompt
    assert "Horário: Desconhecido" not in prompt
    assert "Local: Desconhecido" not in prompt

    # Agora os fatos reais precisam estar la.
    assert caso["vitima"]["nome"] in prompt
    assert caso["papel_do_jogador"]["nome"] in prompt
    for evidencia in caso["evidencias_conhecidas"]:
        assert evidencia["descricao"] in prompt


def test_prompt_separa_verdade_de_evidencia():
    prompt = _orchestrator().build_prompt("sess_prompt", "Nao sei de nada.", get_case())

    assert "VERDADE DO CASO (CONFIDENCIAL — NUNCA REVELE)" in prompt
    assert "EVIDÊNCIAS EM SUAS MÃOS (pode citar)" in prompt
    # A instrucao de nao vazar precisa sobreviver a qualquer refatoracao do prompt.
    assert "NÃO pode citá-la" in prompt


def test_prompt_sem_caso_nao_acusa():
    """Se a pasta de casos sumir, o detetive nao pode sair acusando no vazio."""
    prompt = _orchestrator().build_prompt("sess_prompt", "Eu estava em casa.", None)
    assert "NÃO acuse o suspeito de mentir" in prompt


def test_memoria_registra_quem_falou():
    mem = ContextMemoryManager(storage_dir="data")
    sessao = "sess_papeis"

    caminho = mem._session_md_path(sessao)
    if caminho.exists():
        caminho.unlink()

    mem.append_turn(sessao, "Eu sai as 19h.", role="suspeito")
    mem.append_turn(sessao, "As 19h exatas?", role="detetive")

    conteudo = mem.load_session_md(sessao)
    assert "(SUSPEITO) - Eu sai as 19h." in conteudo
    assert "(DETETIVE) - As 19h exatas?" in conteudo

    # E o contexto montado precisa carregar essa marcacao para o modelo.
    contexto = mem.get_prompt_context(sessao)
    assert "(SUSPEITO)" in contexto and "(DETETIVE)" in contexto


def test_dedup_do_resumo_ainda_funciona_com_papel():
    """
    O summarize deduplica pelo texto da fala. Com o marcador de papel no meio,
    a busca antiga por '] - ' parou de casar — este teste trava a correcao.
    """
    mem = ContextMemoryManager(storage_dir="data")
    sessao = "sess_dedup"

    caminho = mem._session_md_path(sessao)
    if caminho.exists():
        caminho.unlink()

    for _ in range(3):
        mem.append_turn(sessao, "Repetido.", role="suspeito")

    resumo = mem.summarize(sessao, max_lines=5)
    assert resumo.count("Repetido.") == 1
