"""
Provedores de LLM (padrao Strategy).

A regra de negocio nao sabe qual modelo esta atendendo: ela recebe um
LLMProvider e chama `generate(prompt)`. Trocar Gemini por Ollama e mudar a
variavel de ambiente LLM_TYPE, nada mais.

O prompt e montado num lugar so — o PromptOrchestrator — e chega aqui pronto.
Provedor cuida de transporte, nunca de conteudo: se cada um montasse o proprio
prompt, Gemini e Ollama passariam a divergir sem ninguem perceber.
"""
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

from app.services.response_parser import parse_llm_json

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL_PADRAO = "models/gemini-2.5-flash"
OLLAMA_URL_PADRAO = "http://localhost:11434"
OLLAMA_MODELO_PADRAO = "detetive-guilty"
# Teto de 30s do lado do Unity (ApiClient.timeoutSeconds): passar muito disso
# so troca um erro de timeout por outro, mas modelo local frio precisa de folga.
OLLAMA_TIMEOUT_PADRAO = 60.0


class LLMUnavailableError(RuntimeError):
    """
    O provedor esta fora do ar (nao respondeu / recusou conexao).

    Vira HTTP 503 na rota. Distinto de LLMGenerationError de proposito: aqui
    nao adianta tentar de novo no mesmo turno, o servico simplesmente nao esta
    la — e o jogador precisa saber disso em vez de receber um mock silencioso.
    """


class LLMGenerationError(RuntimeError):
    """
    O provedor respondeu, mas a resposta nao serve (erro de API, cota, texto
    fora do contrato). E recuperavel: o orquestrador tenta de novo e, se
    insistir em falhar, cai na resposta de fallback dele.
    """


def _e_erro_de_conexao(exc: BaseException) -> bool:
    """
    Separa 'servico fora do ar' de 'servico respondeu algo ruim'.

    O SDK do Gemini embrulha erros de rede em tipos proprios, entao alem de
    checar as classes conhecidas olhamos o texto — impreciso, mas o custo de
    errar e apenas classificar como 503 algo que seria fallback.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.ReadTimeout, ConnectionError, TimeoutError)):
        return True

    marcadores = ("connection refused", "failed to establish", "name or service not known",
                  "temporary failure in name resolution", "connection error",
                  "max retries exceeded", "timed out", "unreachable")
    texto = f"{type(exc).__name__}: {exc}".lower()
    return any(m in texto for m in marcadores)


class LLMProvider(ABC):
    """Interface que a regra de negocio enxerga."""

    nome: str = "abstrato"

    # Se o provedor aguenta receber a secao VERDADE DO CASO (confidencial) e
    # NAO repeti-la para o jogador.
    #
    # Medido no Llama 3.2 3B local, 8 amostras com o prompt real: vazou a
    # verdade em 5 delas, incluindo dizer o nome do culpado ao suspeito. Nao e
    # problema de redacao do prompt — o modelo nao tem capacidade de seguir a
    # proibicao. A unica correcao confiavel e nao mandar a secao.
    #
    # Consequencia aceita: sem a verdade, o provedor local so detecta
    # contradicao contra EVIDENCIAS e contra falas anteriores do proprio
    # suspeito. Detecta menos — mas nao estraga a partida entregando o final.
    suporta_contexto_confidencial: bool = True

    @abstractmethod
    async def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Recebe o prompt pronto e devolve o dicionario do contrato.

        Levanta LLMUnavailableError se o servico estiver fora do ar, e
        LLMGenerationError se ele responder algo inaproveitavel.
        """

    async def aclose(self) -> None:
        """Libera conexoes. Sem efeito em provedores que nao abrem socket."""


class MockProvider(LLMProvider):
    """
    Respostas deterministicas, sem rede.

    Saiu de dentro do GeminiClient para ca. Enquanto morava la, 'mock' era um
    modo escondido do provedor Gemini — o Ollama nao teria como oferecer o
    mesmo, e testar a regra de jogo exigia falar de Gemini sem necessidade.
    """

    nome = "mock"

    @staticmethod
    def _extrair_fala_do_jogador(prompt: str) -> str:
        """
        Pesca a fala do jogador de dentro do prompt montado.

        Mantido igual ao comportamento anterior porque os testes dependem
        dele: 'Eu nao estava na biblioteca.' precisa continuar disparando
        deteccao de mentira.
        """
        try:
            marcador = "FALA DO JOGADOR:"
            idx = prompt.upper().rfind(marcador)
            if idx == -1:
                return ""

            resto = prompt[idx + len(marcador):]
            if '"' not in resto:
                return resto.strip()

            aspas_1 = resto.find('"')
            aspas_2 = resto.find('"', aspas_1 + 1)
            if aspas_1 != -1 and aspas_2 != -1:
                return resto[aspas_1 + 1:aspas_2].strip()
            return resto.strip()
        except Exception:
            return ""

    async def generate(self, prompt: str) -> Dict[str, Any]:
        texto = self._extrair_fala_do_jogador(prompt).lower()

        if "nao" in texto or "não" in texto or "nunca" in texto:
            return {
                "id_turno": 1,
                "texto_detetive": "Isso contradiz as evidencias. Explique-se melhor.",
                "status_investigacao": {
                    "nivel_suspeita": 80,
                    "congelar_input": False,
                    "detectou_mentira": True,
                    "fim_de_jogo": False,
                },
                "feedback_visual": {
                    "cor_iluminacao": "#FF0000",
                    "bpm_musica": 140,
                    "animacao_trigger": "Intense_Accusation",
                },
            }

        return {
            "id_turno": 1,
            "texto_detetive": "Continue sua declaracao.",
            "status_investigacao": {
                "nivel_suspeita": 20,
                "congelar_input": False,
                "detectou_mentira": False,
                "fim_de_jogo": False,
            },
            "feedback_visual": {
                "cor_iluminacao": "#FFFFFF",
                "bpm_musica": 90,
                "animacao_trigger": "Neutral",
            },
        }


class GeminiProvider(LLMProvider):
    """
    Gemini via SDK oficial, na superficie ASSINCRONA (`client.aio`).

    A versao anterior usava `client.models.generate_content`, que e bloqueante,
    chamada de dentro de uma rota `async def` — cada pergunta congelava o event
    loop inteiro pelos segundos da inferencia. Com um jogador nao aparecia; com
    dois, as requisicoes passavam a se enfileirar.
    """

    nome = "gemini"

    def __init__(self, api_key: str, model: Optional[str] = None):
        import google.genai as genai

        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL", GEMINI_MODEL_PADRAO)
        self.client = genai.Client(api_key=api_key)

    async def generate(self, prompt: str) -> Dict[str, Any]:
        try:
            resposta = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as exc:
            if _e_erro_de_conexao(exc):
                logger.error(f"Gemini inacessivel: {exc}")
                raise LLMUnavailableError(
                    "Nao foi possivel alcancar a API do Gemini."
                ) from exc

            logger.error(f"Gemini respondeu com erro: {exc}")
            raise LLMGenerationError(str(exc)) from exc

        texto = getattr(resposta, "text", None)
        if not texto:
            raise LLMGenerationError("Gemini devolveu resposta vazia.")

        logger.info(f"Resposta bruta (gemini): {texto}")
        return parse_llm_json(texto)


class OllamaProvider(LLMProvider):
    """
    Modelo local via API do Ollama (`/api/chat`).

    Reusa um unico httpx.AsyncClient em vez de abrir um por requisicao: o
    handshake TCP a cada turno e desperdicio num servico que roda na mesma
    maquina. Fechado no shutdown do FastAPI, via aclose().
    """

    nome = "local"

    # Ver a nota em LLMProvider: o 3B vazou a verdade em 5 de 8 amostras.
    suporta_contexto_confidencial = False

    def __init__(self, base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[float] = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", OLLAMA_URL_PADRAO)).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODELO_PADRAO)
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT_PADRAO))
        self._client: Optional[httpx.AsyncClient] = None

    def _obter_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Pede JSON no nivel do runtime. Sem isso, modelos menores gostam de
            # embrulhar a resposta em prosa e todo turno vira retry.
            "format": "json",
        }

        try:
            resposta = await self._obter_client().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            logger.error(f"Ollama inacessivel em {self.base_url}: {exc}")
            raise LLMUnavailableError(
                f"O modelo local nao esta respondendo em {self.base_url}."
            ) from exc

        if resposta.status_code == 404:
            # 404 aqui quase sempre e modelo inexistente, nao rota errada —
            # mensagem generica mandaria o aluno depurar o lugar errado.
            raise LLMUnavailableError(
                f"O Ollama esta no ar, mas o modelo '{self.model}' nao existe. "
                f"Rode: ollama pull {self.model}"
            )

        if resposta.status_code >= 500:
            raise LLMUnavailableError(
                f"Ollama respondeu {resposta.status_code}."
            )

        if resposta.status_code >= 400:
            raise LLMGenerationError(
                f"Ollama recusou a requisicao ({resposta.status_code}): {resposta.text[:200]}"
            )

        try:
            corpo = resposta.json()
        except ValueError as exc:
            raise LLMGenerationError("Ollama devolveu corpo nao-JSON.") from exc

        texto = (corpo.get("message") or {}).get("content", "")
        if not texto:
            raise LLMGenerationError("Ollama devolveu mensagem vazia.")

        logger.info(f"Resposta bruta (ollama): {texto}")
        return parse_llm_json(texto)


@lru_cache(maxsize=None)
def get_llm_provider() -> LLMProvider:
    """
    Factory: le LLM_TYPE e devolve o provedor.

    Cacheado porque o provedor guarda conexao (o Ollama reusa um AsyncClient) e
    porque instanciar o SDK do Gemini a cada requisicao seria desperdicio. Efeito
    colateral: mudar LLM_TYPE exige reiniciar o servidor.

    Usada como dependencia do FastAPI (`Depends(get_llm_provider)`).
    """
    tipo = os.getenv("LLM_TYPE", "gemini").strip().lower()

    if tipo == "local":
        provider = OllamaProvider()
        logger.info(f"LLM_TYPE=local -> Ollama ({provider.model} em {provider.base_url})")
        return provider

    if tipo == "mock":
        logger.info("LLM_TYPE=mock -> respostas deterministicas, sem rede")
        return MockProvider()

    if tipo != "gemini":
        logger.warning(f"LLM_TYPE='{tipo}' desconhecido; usando 'gemini'.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Preserva o comportamento antigo: sem chave, o projeto sempre rodou em
        # mock em vez de subir quebrado. Util para clonar e rodar sem segredo.
        logger.warning("GEMINI_API_KEY ausente; caindo para o provedor mock.")
        return MockProvider()

    provider = GeminiProvider(api_key=api_key)
    logger.info(f"LLM_TYPE=gemini -> {provider.model}")
    return provider
