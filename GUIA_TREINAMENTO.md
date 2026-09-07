# Guia — treinar o detetive local e ligar no Ollama

Do zero ao modelo rodando no jogo. Tempo total: **~1h**, sendo ~40 min de espera.

| Arquivo | O que é |
|---|---|
| `dataset_interrogatorio.jsonl` | 50 exemplos de treino, formato chat |
| `treinamento_colab.py` | Pipeline QLoRA em 9 células de Colab |
| `GUIA_TREINAMENTO.md` | Este arquivo |

---

## Antes de começar: por que o alvo de treino é JSON

O `OllamaProvider` do backend envia `format: "json"` e passa a resposta pelo
`parse_llm_json`, que espera o `ResponseContract`. Um modelo treinado para
responder em prosa faria **todo turno cair no fallback do parser**, que carimba
`nivel_suspeita: 50` fixo e detecta mentira por palavra-chave — o fine-tuning
inteiro seria descartado na última etapa.

Por isso o campo `assistant` do dataset é o objeto JSON do contrato, com a fala
incisiva do detetive dentro de `texto_detetive`. Os 50 exemplos foram validados
contra o `ResponseContract` real do projeto: **50/50 passam**.

---

## Passo 1 — Abrir o Colab com GPU

1. Vá em [colab.research.google.com](https://colab.research.google.com) e crie um notebook.
2. **Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU → Salvar.**

Sem isso o treino não roda — o Unsloth exige CUDA. Confirme com uma célula:

```python
!nvidia-smi
```

Deve aparecer `Tesla T4`.

## Passo 2 — Subir os dois arquivos

No painel lateral esquerdo, ícone de pasta → botão de upload. Envie:

- `dataset_interrogatorio.jsonl`
- `treinamento_colab.py` (opcional — serve como referência para copiar as células)

> O `.jsonl` precisa ficar na raiz (`/content/`), porque a célula 4 o carrega por
> caminho relativo. Se você criar uma subpasta, ajuste o `data_files`.

**Atenção:** o Colab apaga os arquivos quando a sessão desconecta. Se a sessão
cair, refaça o upload antes de recomeçar.

## Passo 3 — Rodar as células, em ordem

Abra `treinamento_colab.py`, e copie cada bloco marcado `# %% CELULA N` para uma
célula do notebook. Rode na ordem.

| Célula | O que faz | Tempo |
|---|---|---|
| 1 | Instala Unsloth e dependências | ~3 min |
| 2 | Baixa o Llama-3.2-3B-Instruct em 4 bits | ~3 min |
| 3 | Adiciona os adaptadores LoRA | segundos |
| 4 | Carrega e formata o dataset | segundos |
| 5 | Configura o `SFTTrainer` | segundos |
| 6 | **Treina** | ~8 min |
| 7 | Testa a saída antes de exportar | ~1 min |
| 8 | **Exporta em GGUF q4_k_m** | ~20 min |
| 9 | Baixa o `.gguf` | ~5 min |

**A célula 7 existe para você não perder 20 minutos à toa.** Ela gera uma
resposta e tenta fazer `json.loads` nela. Se não for JSON válido ali, não vai ser
depois de quantizar — volte à célula 6 e treine mais uma ou duas épocas
(`num_train_epochs = 6`) antes de exportar.

Na célula 6, a saída esperada é a *loss* caindo de algo em torno de 2,0 para
abaixo de 0,5 ao longo de ~24 passos. Se ela ficar parada perto de 2,0, o
dataset não está sendo lido — confira o `print` de exemplos da célula 4.

## Passo 4 — Baixar o modelo

A célula 9 baixa `modelo_detetive/unsloth.Q4_K_M.gguf`, cerca de **2 GB**.

O download pelo widget do Colab é lento e cai com facilidade. Se falhar, use o
Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp modelo_detetive/*.gguf /content/drive/MyDrive/
```

E baixe do Drive pelo navegador.

---

## Passo 5 — Registrar no Ollama

O `.gguf` sozinho não basta: o Ollama precisa de um **Modelfile** que diga qual
template de chat usar e qual system prompt aplicar.

Coloque o `.gguf` numa pasta e crie ao lado um arquivo chamado `Modelfile`
(sem extensão):

```
FROM ./unsloth.Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"

SYSTEM """Você é um detetive de homicídios experiente, cínico e paciente, conduzindo um interrogatório. Confronte a fala do suspeito com as evidências e com o que ele mesmo já disse. Responda SEMPRE com um único objeto JSON válido, sem markdown."""
```

O `num_ctx 4096` importa: o prompt real do backend tem ~100 linhas (fatos do
caso, evidências, histórico). Com o padrão de 2048 o começo do prompt seria
cortado — e é justamente ali que ficam os fatos do crime.

Depois, no terminal, dentro dessa pasta:

```bash
ollama create detetive-guilty -f Modelfile
```

**O nome tem que ser exatamente `detetive-guilty`** — é o que o
`OllamaProvider` procura (`OLLAMA_MODEL`, com esse padrão).

Confira:

```bash
ollama list
```

## Passo 6 — Ligar no backend

No `.env` do backend:

```
LLM_TYPE=local
```

Reinicie o servidor — a factory é cacheada, então trocar `LLM_TYPE` com o
servidor de pé não faz efeito.

Teste sem abrir o Unity:

```bash
curl -X POST http://localhost:8000/interrogate -H "Content-Type: application/json" -d "{\"session_id\":\"teste\",\"player_text\":\"Eu nao estava la naquela noite.\"}"
```

Resposta esperada: o JSON do contrato, com `texto_detetive` preenchido.

---

## Se der errado

| Sintoma | Causa provável |
|---|---|
| `HTTP 503` no backend | Ollama não está rodando. Suba com `ollama serve`. |
| 503 dizendo que o modelo não existe | O `ollama create` usou outro nome. Refaça com `detetive-guilty`. |
| Respostas em prosa, `nivel_suspeita` sempre 50 | O modelo não aprendeu o formato JSON — caiu no fallback do parser. Treine mais épocas. |
| Detetive acusa de mentira o tempo todo | Overfitting nos exemplos de contradição. Reduza para 2-3 épocas. |
| `CUDA out of memory` na célula 6 | Ambiente sem T4, ou outra sessão ocupando a GPU. Reinicie o ambiente de execução. |
| Célula 8 travada há mais de 40 min | Ela compila o llama.cpp do zero. Se passar de 1h, reinicie e rode de novo. |
| Timeout no Unity com o modelo local | O `ApiClient` desiste em 30 s. Suba `timeoutSeconds` no Inspector, ou use `OLLAMA_TIMEOUT` menor para falhar antes. |

---

## Sobre o dataset

50 exemplos, distribuídos em quatro tipos de furo investigativo:

- **13** — contradições de horário e cronologia
- **13** — falsos álibis desmentidos por câmera, vizinho ou recibo
- **12** — objetos da cena (digitais, ferramentas, roupas, calçado)
- **12** — relacionamento forçado com a vítima ou testemunhas

Dois cuidados de balanceamento que valem ser ditos na defesa:

**Metade não é mentira** (26 contradições / 24 falas legítimas). Um dataset onde
quase tudo é mentira ensina o modelo a acusar sempre — em jogo, o detetive
acusaria o jogador a cada frase e a barra de suspeita viveria no máximo.

**A suspeita acompanha o número do turno.** Mentira no turno 2 sobe menos que
mentira no turno 7, porque suspeita acumula. Sem isso o modelo aprende a saltar
direto para 85 no primeiro deslize.

Distribuição resultante: 7 exemplos de suspeita baixa (≤30), 19 média (31-60),
24 alta (>60); média 55.

Dois campos do contrato — `congelar_input` e `fim_de_jogo` — são reescritos pelo
`PromptOrchestrator` depois da resposta do modelo. Eles estão coerentes no
dataset, mas quem manda neles é o servidor.

### Para ampliar o dataset

Os 50 exemplos são o mínimo defensável. Se quiser mais, mantenha as proporções
acima e valide antes de treinar — JSONL quebrado só aparece no meio do treino:

```python
import json
with open("dataset_interrogatorio.jsonl", encoding="utf-8") as f:
    for i, linha in enumerate(f, 1):
        obj = json.loads(linha)                      # estoura se inválido
        assert [m["role"] for m in obj["messages"]] == ["system", "user", "assistant"]
        json.loads(obj["messages"][2]["content"])    # o alvo tem que ser JSON
print(f"{i} linhas válidas")
```
