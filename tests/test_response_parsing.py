from app.services.prompt_orchestrator import PromptOrchestrator
from app.services.context_memory import ContextMemoryManager
from app.schemas import ResponseContract


def test_prompt_analyze_returns_contract_like():
    memory = ContextMemoryManager(storage_dir="data")
    orchestrator = PromptOrchestrator(memory=memory)
    # Modo mock garantido quando não há GEMINI_API_KEY
    result = orchestrator.analyze("sess_test", "Eu nunca estive lá.", mode='mock')

    # Deve possuir campos básicos do contrato
    assert isinstance(result.get('id_turno'), int)
    assert isinstance(result.get('texto_detetive'), str)
    status = result.get('status_investigacao')
    assert isinstance(status.get('nivel_suspeita'), int)

    # Validar contra o modelo Pydantic
    # Isto deve lançar erro se estiver fora do contrato
    ResponseContract(**result)
