import json
import os
import time
from typing import Dict, Any
from app.services.gemini_client import GeminiClient
from app.schemas import ResponseContract
from pydantic import ValidationError

class PromptOrchestrator:
    def __init__(self, memory):
        self.memory = memory
        self.gemini_client = GeminiClient()
        self.turn_counter = {}
        self.crime_facts = self._load_crime_facts()
        self.exemplos = self._get_few_shot_examples()

    def _load_crime_facts(self) -> Dict[str, Any]:
        caminho = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fatos_crime.json')
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Aviso: arquivo de fatos nao encontrado em {caminho}")
            return {"victim": "Desconhecida", "time": "Desconhecido", "location": "Desconhecido"}

    def _get_few_shot_examples(self) -> str:
        return """
Exemplo 1 - Contradição Factual:
Contexto: A vítima foi encontrada morta às 20:00 na biblioteca.
Jogador: "Eu não estava na biblioteca ontem à noite."
Análise: O jogador nega estar no local do crime, mas isso contradiz evidências. Mentira detectada.

Exemplo 2 - Continuidade Quebrada:
Contexto: O jogador disse anteriormente que estava sozinho em casa.
Jogador: "Meu amigo João estava comigo a noite toda."
Análise: Esta declaração quebra a continuidade da história anterior. Possível mentira.

Exemplo 3 - Resposta Neutra:
Contexto: Pergunta sobre o paradeiro da vítima.
Jogador: "Eu vi Pedro saindo da biblioteca por volta das 19:30."
Análise: Esta informação não contradiz fatos conhecidos. Resposta neutra.
"""

    def build_prompt(self, session_id: str, player_text: str) -> str:
        contexto = self.memory.get_prompt_context(session_id)

        vitima = self.crime_facts.get('victim', 'Desconhecida')
        hora = self.crime_facts.get('time', 'Desconhecido')
        local = self.crime_facts.get('location', 'Desconhecido')

        prompt = f"""Você é um detetive experiente analisando depoimentos em um jogo de investigação criminal.
Sua tarefa é analisar a fala do jogador, detectar possíveis mentiras ou contradições, e gerar uma resposta apropriada.

FATOS CONHECIDOS DO CRIME:
- Vítima: {vitima}
- Horário: {hora}
- Local: {local}

{contexto}

EXEMPLOS DE ANÁLISE:
{self.exemplos}

INSTRUÇÕES:
1. Analise a fala do jogador procurando contradições com os fatos conhecidos
2. Considere o contexto da sessão anterior
3. Detecte mentiras, inconsistências ou continuidade quebrada
4. Responda como um detetive: pressione por mais informações, confronte mentiras
5. Retorne APENAS um objeto JSON válido com os campos especificados, sem blocos de código markdown ou formatação adicional

FALA DO JOGADOR: "{player_text}"

Responda com JSON puro (sem ```json):
{{
  "id_turno": 1,
  "texto_detetive": "Sua resposta como detetive",
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
        return prompt

    def analyze(self, session_id: str, player_text: str, mode: str = 'mock') -> Dict[str, Any]:
        if session_id not in self.turn_counter:
            self.turn_counter[session_id] = 0
            
        self.turn_counter[session_id] += 1

        prompt = self.build_prompt(session_id, player_text)

        tentativas = 0
        resultado = None

        # tenta mandar pro gemini e faz retry se o json vier quebrado
        while tentativas <= 2:
            res = self.gemini_client.send_prompt(prompt, mode=mode)
            
            if type(res) is dict and 'parsed' in res:
                dados = res['parsed']
                
                try:
                    # ajusta o turno antes de validar
                    dados['id_turno'] = self.turn_counter[session_id]
                    ResponseContract.model_validate(dados)
                    resultado = dados
                    break
                except Exception:
                    tentativas += 1
                    time.sleep(0.1)
            else:
                tentativas += 1
                time.sleep(0.1)

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