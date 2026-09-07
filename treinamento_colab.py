# -*- coding: utf-8 -*-
"""
Fine-tuning QLoRA do detetive do Guilty — Llama-3.2-3B-Instruct via Unsloth.

Feito para rodar no Google Colab com GPU T4 (16 GB). Cada bloco marcado com
`# %% CELULA N` e uma celula do notebook: copie e cole na ordem.

Saida final: modelo_detetive/unsloth.Q4_K_M.gguf, pronto para o Ollama.

--------------------------------------------------------------------------
POR QUE O ALVO DE TREINO E JSON, E NAO PROSA
--------------------------------------------------------------------------
O backend do Guilty (OllamaProvider) envia `format: "json"` e passa a resposta
por parse_llm_json, que espera o ResponseContract. Se o modelo fosse treinado
para responder em prosa, TODO turno cairia no fallback do parser — que carimba
nivel_suspeita=50 fixo e detecta mentira por palavra-chave. O fine-tuning seria
descartado na ultima etapa do pipeline.

Por isso o campo `assistant` do dataset e o objeto JSON do contrato, com a fala
incisiva do detetive dentro de `texto_detetive`.
"""

# %% CELULA 1 — Instalacao
# A T4 do Colab e Turing (sm_75). O Unsloth resolve as dependencias certas
# sozinho; as versoes do TRL/transformers mudam com frequencia e quebram a API
# do SFTTrainer, entao deixamos o instalador do Unsloth mandar.
#
#   !pip install -q unsloth
#   !pip install -q --no-deps --upgrade "trl<0.16.0" peft accelerate bitsandbytes
#
# Se algo quebrar na importacao, o notebook oficial do Unsloth para Llama 3.2
# tem sempre a linha de instalacao mais atual — use a de la e volte para a
# celula 2.

# %% CELULA 2 — Carregar o modelo base em 4 bits
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 2048   # nossos exemplos ficam em ~700 chars; 2048 tokens sobra
DTYPE = None            # None = o Unsloth detecta (float16 na T4)
LOAD_IN_4BIT = True     # QLoRA: sem isso, 3B nao cabe com folga nos 16 GB

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

# %% CELULA 3 — Adaptadores LoRA
# r=16 e o meio-termo usual: r baixo demais nao aprende o formato JSON, r alto
# demais decora 50 exemplos e perde a capacidade de generalizar para falas novas.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha     = 16,
    lora_dropout   = 0,      # 0 e o caminho otimizado do Unsloth
    bias           = "none", # idem
    use_gradient_checkpointing = "unsloth",  # corta VRAM; essencial na T4
    random_state   = 3407,
    use_rslora     = False,
    loftq_config   = None,
)

# %% CELULA 4 — Carregar e formatar o dataset
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Llama 3.2 usa o mesmo chat template da 3.1.
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

dataset = load_dataset(
    "json",
    data_files = "dataset_interrogatorio.jsonl",
    split = "train",
)
print(f"exemplos carregados: {len(dataset)}")


def formatar(exemplos):
    """Aplica o chat template do Llama nas conversas system/user/assistant."""
    textos = [
        tokenizer.apply_chat_template(msgs, tokenize = False, add_generation_prompt = False)
        for msgs in exemplos["messages"]
    ]
    return {"text": textos}


dataset = dataset.map(formatar, batched = True)
print(dataset[0]["text"][:600])

# %% CELULA 5 — Configurar o treinador
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# batch 2 x acumulo 4 = batch efetivo 8. A T4 nao aguenta batch real maior com
# 2048 de contexto; o acumulo da o mesmo efeito de gradiente gastando VRAM de um.
trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = dataset,
    dataset_text_field = "text",
    max_seq_length    = MAX_SEQ_LENGTH,
    dataset_num_proc  = 2,
    packing           = False,   # exemplos curtos e independentes; packing atrapalharia
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps      = 5,
        # 50 exemplos e pouco: 4 epocas dao ~24 passos, suficiente para fixar o
        # formato JSON e o tom. Subir muito acima disso comeca a decorar as 50
        # respostas em vez de aprender o padrao.
        num_train_epochs  = 4,
        learning_rate     = 2e-4,
        fp16              = not is_bfloat16_supported(),  # T4 -> fp16
        bf16              = is_bfloat16_supported(),      # A100/L4 -> bf16
        logging_steps     = 1,
        optim             = "adamw_8bit",
        weight_decay      = 0.01,
        lr_scheduler_type = "linear",
        seed              = 3407,
        output_dir        = "outputs",
        report_to         = "none",
    ),
)

# Treina a perda APENAS nos turnos do assistente. Sem isto o modelo tambem
# aprende a reproduzir o system prompt e a fala do suspeito — desperdicio de
# capacidade e piora a qualidade da resposta.
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# %% CELULA 6 — Treinar
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} | {round(gpu.total_memory / 1024**3, 2)} GB")

stats = trainer.train()

print(f"tempo: {round(stats.metrics['train_runtime'] / 60, 2)} min")
print(f"pico de VRAM: {round(torch.cuda.max_memory_reserved() / 1024**3, 2)} GB")

# %% CELULA 7 — Testar antes de exportar
# Vale gastar dois minutos aqui: se a saida nao for JSON valido agora, nao vai
# ser depois de quantizar, e voce evita 20 minutos de export inutil.
import json

FastLanguageModel.for_inference(model)

SYSTEM_TESTE = (
    "Você é um detetive de homicídios experiente, cínico e paciente, conduzindo um "
    "interrogatório. Seu trabalho é confrontar a fala do suspeito com as evidências "
    "e com o que ele mesmo já disse, procurando furos lógicos, temporais e de álibi.\n"
    "Regras: cite apenas o que está em EVIDÊNCIAS; não invente provas; não acuse sem "
    "base — não ter álibi não é mentira; pressione por horário, local e quem confirma.\n"
    "Responda SEMPRE com um único objeto JSON válido, sem markdown e sem texto fora "
    "do JSON, neste formato:\n"
    '{"id_turno": int, "texto_detetive": str, "status_investigacao": {"nivel_suspeita": '
    '0-100, "congelar_input": bool, "detectou_mentira": bool, "fim_de_jogo": bool}, '
    '"feedback_visual": {"cor_iluminacao": "#RRGGBB", "bpm_musica": 60-140, '
    '"animacao_trigger": str}}'
)

USER_TESTE = (
    "EVIDÊNCIAS:\n"
    "- Câmera do corredor registra o suspeito às 20:11 no andar do depósito.\n"
    "\n"
    "HISTÓRICO DO INTERROGATÓRIO:\n"
    "- Suspeito disse que nunca subiu àquele andar.\n"
    "\n"
    'FALA DO SUSPEITO: "Eu já falei, não subi lá em nenhum momento."'
)

entrada = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_TESTE},
     {"role": "user",   "content": USER_TESTE}],
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

saida = model.generate(input_ids = entrada, max_new_tokens = 300,
                       temperature = 0.7, do_sample = True)
texto = tokenizer.decode(saida[0][entrada.shape[1]:], skip_special_tokens = True)

print(texto)
try:
    dados = json.loads(texto)
    faltando = {"id_turno", "texto_detetive", "status_investigacao",
                "feedback_visual"} - set(dados)
    print("\nJSON valido." if not faltando else f"\nJSON valido, mas faltam: {faltando}")
except json.JSONDecodeError as e:
    print(f"\nATENCAO: saida nao e JSON valido ({e}). "
          "Treine mais uma epoca antes de exportar.")

# %% CELULA 8 — Exportar em GGUF quantizado
# Demora ~15-25 min na T4: compila o llama.cpp, converte para f16 e quantiza.
# Nao feche a aba do navegador durante essa celula.
model.save_pretrained_gguf(
    "modelo_detetive",
    tokenizer,
    quantization_method = "q4_k_m",
)

# %% CELULA 9 — Baixar o .gguf
# Um 3B em q4_k_m fica em ~2 GB. O download pelo widget do Colab e lento e cai
# com facilidade; se der problema, monte o Drive e copie para la:
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   !cp modelo_detetive/*.gguf /content/drive/MyDrive/
#
import os

for arquivo in os.listdir("modelo_detetive"):
    if arquivo.endswith(".gguf"):
        caminho = os.path.join("modelo_detetive", arquivo)
        print(f"{arquivo} — {round(os.path.getsize(caminho) / 1024**3, 2)} GB")

from google.colab import files

files.download("modelo_detetive/unsloth.Q4_K_M.gguf")
