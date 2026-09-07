import asyncio

from app.services.prompt_orchestrator import PromptOrchestrator
from app.services.context_memory import ContextMemoryManager
from app.services.llm_providers import MockProvider
from app.schemas import ResponseContract


def test_prompt_analyze_returns_contract_like():
    memory = ContextMemoryManager(storage_dir="data")
    orchestrator = PromptOrchestrator(memory=memory)
    # O provedor entra por parametro: o teste nao depende mais de variavel de
    # ambiente nem de haver (ou nao) uma GEMINI_API_KEY na maquina.
    result = asyncio.run(
        orchestrator.analyze("sess_test", "Eu nunca estive lá.", provider=MockProvider())
    )

    # Deve possuir campos básicos do contrato
    assert isinstance(result.get('id_turno'), int)
    assert isinstance(result.get('texto_detetive'), str)
    status = result.get('status_investigacao')
    assert isinstance(status.get('nivel_suspeita'), int)

    # Validar contra o modelo Pydantic
    # Isto deve lançar erro se estiver fora do contrato
    ResponseContract(**result)
