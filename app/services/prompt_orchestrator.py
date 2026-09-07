import asyncio
import json
import logging
from typing import Dict, Any, Optional

from app.schemas import ResponseContract
from app.services.case_repository import get_case
from app.services.llm_providers import LLMGenerationError, LLMProvider

logger = logging.getLogger(__name__)

class PromptOrchestrator:
    def __init__(self, memory):
        self.memory = memory
        self.turn_counter = {}
        self.exemplos = self._get_few_shot_examples()

    def _get_few_shot_examples(self) -> str:
        """
        Exemplos deliberadamente ABSTRATOS.

        A versão anterior citava uma biblioteca e um horário de 20:00. Como o
        arquivo de fatos nunca existiu, esses detalhes inventados eram a única
        informação concreta de crime no prompt inteiro — e o modelo passava a
        tratá-los como o caso real, improvisando em cima deles. Com fatos de
        verdade no prompt, os exemplos voltam a fazer só o trabalho deles:
        ensinar o formato do raciocínio, sem competir com o caso.
        """
        return """
Exemplo A - Contradição com evidência:
Uma evidência em suas mãos afirma X. O suspeito afirma algo incompatível com X.
Análise: contradição direta. detectou_mentira = true, suspeita sobe muito.

Exemplo B - Contradição com ele mesmo:
O suspeito afirmou antes que estava sozinho. Agora afirma que havia alguém com ele.
Análise: a história mudou. detectou_mentira = true, suspeita sobe.

Exemplo C - Lacuna preenchida de forma plausível:
O suspeito dá um detalhe verificável que explica um período em aberto e não colide
com nenhuma evidência.
Análise: não é mentira. detectou_mentira = false, suspeita desce.

Exemplo D - Evasiva:
O suspeito desconversa ou se recusa a responder.
Análise: não é mentira, mas não colabora. detectou_mentira = false, suspeita sobe pouco.
"""

    # ─── Blocos do prompt montados a partir do caso ──────────────────────────

    def _bloco_identidade(self, caso: Dict[str, Any]) -> str:
        papel = caso.get('papel_do_jogador', {})
        vitima = caso.get('vitima', {})
        return f"""CASO: {caso.get('titulo', 'Sem título')}
VÍTIMA: {vitima.get('nome', 'Desconhecida')}, {vitima.get('idade', '?')} anos — {vitima.get('funcao', '')}

QUEM ESTÁ NA SUA FRENTE:
- Nome: {papel.get('nome', 'Desconhecido')}
- Função: {papel.get('funcao', '')}
- Situação: {papel.get('situacao', '')}
- {papel.get('instrucao_ao_detetive', '')}"""

    def _bloco_verdade(self, caso: Dict[str, Any]) -> str:
        """
        A verdade do caso. Fica no prompt para o modelo conseguir JULGAR as
        falas, com proibição explícita de revelá-la — é o que separa um detetive
        que percebe a mentira de um que só desconfia de tudo.
        """
        verdade = caso.get('verdade', {})
        linha = "\n".join(
            f"  {item.get('hora', '??')} — {item.get('fato', '')}"
            for item in verdade.get('linha_do_tempo', [])
        )
        culpado_e_o_jogador = verdade.get('jogador_e_culpado')

        return f"""=== VERDADE DO CASO (CONFIDENCIAL — NUNCA REVELE) ===
Esta seção existe para você AVALIAR o que o suspeito diz. Você NÃO pode citá-la,
insinuá-la nem deixá-la transparecer. Só pode mencionar o que estiver na seção
de EVIDÊNCIAS. Se o suspeito perguntar como você sabe de algo, cite a evidência.

O que aconteceu: {verdade.get('o_que_aconteceu', 'Desconhecido')}
Culpado real: {verdade.get('culpado', 'Desconhecido')}
O suspeito à sua frente é o culpado? {'SIM' if culpado_e_o_jogador else 'NÃO'}

Linha do tempo real:
{linha or '  (não informada)'}"""

    def _bloco_evidencias(self, caso: Dict[str, Any]) -> str:
        itens = "\n".join(
            f"  - [{e.get('fonte', '?')}] {e.get('descricao', '')}"
            for e in caso.get('evidencias_conhecidas', [])
        )
        lacunas = "\n".join(f"  - {l}" for l in caso.get('lacunas', []))

        return f"""=== EVIDÊNCIAS EM SUAS MÃOS (pode citar) ===
{itens or '  (nenhuma)'}

=== O QUE VOCÊ AINDA NÃO SABE (arranque isto dele) ===
{lacunas or '  (nada)'}"""

    def _bloco_suspeita(self, caso: Dict[str, Any]) -> str:
        regras = caso.get('regras_de_suspeita', {})
        linhas = "\n".join(
            f"  - {chave.replace('_', ' ')}: {valor}"
            for chave, valor in regras.items()
            if chave != 'observacao'
        )
        obs = regras.get('observacao', '')

        return f"""=== COMO MEXER NO NÍVEL DE SUSPEITA ===
Parta do nível atual e ajuste:
{linhas or '  (sem regras definidas)'}
{f'  ATENÇÃO: {obs}' if obs else ''}"""

    def build_prompt(self, session_id: str, player_text: str,
                     caso: Optional[Dict[str, Any]] = None,
                     incluir_verdade: bool = True) -> str:
        """
        `incluir_verdade=False` omite a seção confidencial do caso.

        Existe porque modelos pequenos não conseguem sustentar a proibição de
        não revelá-la: medido no Llama 3.2 3B local, com o prompt real, a
        verdade vazou em 5 de 8 respostas — uma delas dizendo ao suspeito o
        nome do culpado. Quem decide é a capability do provedor
        (LLMProvider.suporta_contexto_confidencial), não uma configuração à
        parte, para não existir combinação de env que mande a verdade para um
        modelo que a repete.
        """
        contexto = self.memory.get_prompt_context(session_id)

        if caso is None:
            # Sem caso carregado o jogo continua, mas sem base factual — que é
            # exatamente o estado anterior a esta mudança. Loga alto porque é
            # degradação silenciosa: a partida "funciona" e a detecção não.
            logger.error("Nenhum caso disponível: o detetive vai interrogar sem fatos.")
            return self._build_prompt_sem_caso(contexto, player_text)

        if incluir_verdade:
            bloco_verdade = self._bloco_verdade(caso)
        else:
            bloco_verdade = (
                "=== VOCÊ NÃO TEM A SOLUÇÃO DO CASO ===\n"
                "Você só dispõe das evidências abaixo. Detecte contradição apenas\n"
                "contra elas ou contra o que o próprio suspeito já disse nesta sessão.\n"
                "Não afirme saber onde ele esteve nem quem é o culpado."
            )

        return f"""Você é um detetive experiente conduzindo um interrogatório.

{self._bloco_identidade(caso)}

{bloco_verdade}

{self._bloco_evidencias(caso)}

{contexto}

{self._bloco_suspeita(caso)}

EXEMPLOS DE ANÁLISE:
{self.exemplos}

INSTRUÇÕES:
1. Compare a fala do suspeito com as EVIDÊNCIAS e com a VERDADE DO CASO
2. Compare também com o que ele mesmo já disse nesta sessão
3. Só marque detectou_mentira quando houver colisão real — não ter álibi não é mentira
4. Pressione as lacunas: peça horário, local e quem pode confirmar
5. Nunca revele a VERDADE DO CASO; cite apenas evidências
6. Retorne APENAS um objeto JSON válido, sem blocos de código markdown

FALA DO JOGADOR: "{player_text}"

Responda com JSON puro (sem ```json):
{{
  "id_turno": 1,
  "texto_detetive": "<escreva aqui a sua fala, em segunda pessoa>",
  "status_investigacao": {{
    "nivel_suspeita": 0-100,
    "congelar_input": false,
    "detectou_mentira": true/false,
    "fim_de_jogo": false
  }},
  "feedback_visual": {{
    "cor_iluminacao": "#HEXCODE",
    "bpm_musica": 60-140,
    "animacao_trigger": "NomeDaAnimacao"
  }}
}}
"""

    def _build_prompt_sem_caso(self, contexto: str, player_text: str) -> str:
        """Degradação para quando a pasta de casos está vazia ou corrompida."""
        return f"""Você é um detetive experiente conduzindo um interrogatório.
Não há fatos do caso disponíveis: conduza o interrogatório pedindo detalhes
verificáveis e NÃO acuse o suspeito de mentir, porque você não tem contra o que
comparar.

{contexto}

EXEMPLOS DE ANÁLISE:
{self.exemplos}

FALA DO JOGADOR: "{player_text}"

Responda com JSON puro (sem ```json):
{{
  "id_turno": 1,
  "texto_detetive": "<escreva aqui a sua fala, em segunda pessoa>",
  "status_investigacao": {{
    "nivel_suspeita": 0-100,
    "congelar_input": false,
    "detectou_mentira": false,
    "fim_de_jogo": false
  }},
  "feedback_visual": {{
    "cor_iluminacao": "#HEXCODE",
    "bpm_musica": 60-140,
    "animacao_trigger": "NomeDaAnimacao"
  }}
}}
"""

    async def analyze(self, session_id: str, player_text: str,
                      provider: LLMProvider,
                      case_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Roda um turno de interrogatorio.

        Recebe o provedor por parametro em vez de guardar um no construtor: assim
        a regra de jogo nao sabe qual modelo esta atendendo, e testar nao exige
        mexer em variavel de ambiente — basta passar um MockProvider.

        LLMUnavailableError NAO e capturado aqui de proposito: servico fora do ar
        e problema de infraestrutura, e a rota traduz isso em HTTP 503. Engolir
        aqui devolveria um mock silencioso e o jogador nunca saberia que o
        modelo esta desligado.
        """
        if session_id not in self.turn_counter:
            self.turn_counter[session_id] = 0

        self.turn_counter[session_id] += 1

        caso = get_case(case_id)
        # A capability do provedor decide se a verdade do caso entra no prompt.
        prompt = self.build_prompt(
            session_id, player_text, caso,
            incluir_verdade=provider.suporta_contexto_confidencial,
        )

        tentativas = 0
        resultado = None

        # tenta gerar e faz retry se o json vier fora do contrato
        while tentativas <= 2:
            try:
                dados = await provider.generate(prompt)
            except LLMGenerationError as exc:
                logger.warning(f"Falha recuperavel do provedor {provider.nome}: {exc}")
                tentativas += 1
                await asyncio.sleep(0.1)
                continue

            try:
                # ajusta o turno antes de validar
                dados['id_turno'] = self.turn_counter[session_id]
                ResponseContract.model_validate(dados)
                resultado = dados
                break
            except Exception:
                tentativas += 1
                await asyncio.sleep(0.1)

        # se der erro em todas as tentativas, usa a resposta fallback
        if not resultado:
            return self._fallback_response(session_id, player_text)

        try:
            nivel = int(resultado.get('status_investigacao', {}).get('nivel_suspeita', 0))
        except Exception:
            nivel = 0

        try:
            self.memory.append_suspicion(session_id, nivel)
        except Exception:
            pass

        # trava o input se a suspeita passar de 80
        congelar = True if nivel > 80 else False

        # game over se a suspeita ficar acima de 90 por 3 turnos seguidos
        recentes = self.memory.get_recent_suspicions(session_id, n=3)
        fim = False
        if len(recentes) >= 3 and all(v >= 90 for v in recentes[-3:]):
            fim = True

        status = resultado.get('status_investigacao', {})
        status['congelar_input'] = congelar
        status['fim_de_jogo'] = fim
        resultado['status_investigacao'] = status

        try:
            if status.get('detectou_mentira'):
                self.memory.append_contradiction(session_id, resultado['id_turno'])
        except Exception:
            pass

        return resultado

    def _fallback_response(self, session_id: str, player_text: str) -> Dict[str, Any]:
        turno = self.turn_counter.get(session_id, 1)

        return {
            "id_turno": turno,
            "texto_detetive": f"Interessante... você disse: '{player_text}'. Continue.",
            "status_investigacao": {
                "nivel_suspeita": 30,
                "congelar_input": False,
                "detectou_mentira": False,
                "fim_de_jogo": False
            },
            "feedback_visual": {
                "cor_iluminacao": "#FFFF00",
                "bpm_musica": 100,
                "animacao_trigger": "Thinking"
            }
        }