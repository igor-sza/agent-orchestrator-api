import os
import sys
import json
import logging
from typing import Dict, Any
import google.genai as genai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            print("Chave do Gemini carregada. Modo real ativo.")
        else:
            self.client = None
            print("Sem chave da API. Rodando com mock.")

    def send_prompt(self, prompt: str, mode: str = 'mock') -> Dict[str, Any]:
        if mode == 'real' and self.api_key and self.client:
            try:
                response = self.client.models.generate_content(
                    model='models/gemini-2.5-flash',
                    contents=prompt
                )
                logger.info(f"Resposta bruta: {response.text}")
                return {
                    "raw_response": response.text,
                    "parsed": self._parse_response(response.text)
                }
            except Exception as e:
                print(f"Deu erro na API do Gemini: {e}. Caindo pro mock.")
                return self._mock_response(prompt)
        
        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> Dict[str, Any]:
        player_text = ""
        try:
            marcador = 'FALA DO JOGADOR:'
            idx = prompt.upper().rfind(marcador)
            
            if idx != -1:
                resto = prompt[idx + len(marcador):]
                if '"' in resto:
                    aspas_1 = resto.find('"')
                    aspas_2 = resto.find('"', aspas_1 + 1)
                    if aspas_1 != -1 and aspas_2 != -1:
                        player_text = resto[aspas_1+1:aspas_2].strip()
                    else:
                        player_text = resto.strip()
                else:
                    player_text = resto.strip()
        except Exception:
            pass

        texto_lower = player_text.lower()
        
        if "nao" in texto_lower or "não" in texto_lower or "nunca" in texto_lower:
            return {
                "raw_response": "Mentira detectada",
                "parsed": {
                    "id_turno": 1,
                    "texto_detetive": "Isso contradiz as evidencias. Explique-se melhor.",
                    "status_investigacao": {
                        "nivel_suspeita": 80,
                        "congelar_input": False,
                        "detectou_mentira": True,
                        "fim_de_jogo": False
                    },
                    "feedback_visual": {
                        "cor_iluminacao": "#FF0000",
                        "bpm_musica": 140,
                        "animacao_trigger": "Intense_Accusation"
                    }
                }
            }
            
        return {
            "raw_response": "Resposta neutra",
            "parsed": {
                "id_turno": 1,
                "texto_detetive": "Continue sua declaracao.",
                "status_investigacao": {
                    "nivel_suspeita": 20,
                    "congelar_input": False,
                    "detectou_mentira": False,
                    "fim_de_jogo": False
                },
                "feedback_visual": {
                    "cor_iluminacao": "#FFFFFF",
                    "bpm_musica": 90,
                    "animacao_trigger": "Neutral"
                }
            }
        }

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            if response_text.strip().startswith('```json'):
                try:
                    inicio = response_text.index('```json') + 7
                    fim = response_text.rindex('```')
                    texto_limpo = response_text[inicio:fim].strip()
                    dados = json.loads(texto_limpo)
                    
                    if type(dados) is dict:
                        return dados
                except Exception:
                    pass

        return {
            "id_turno": 1,
            "texto_detetive": response_text.strip(),
            "status_investigacao": {
                "nivel_suspeita": 50,
                "congelar_input": False,
                "detectou_mentira": "mentira" in response_text.lower(),
                "fim_de_jogo": False
            },
            "feedback_visual": {
                "cor_iluminacao": "#FFFF00",
                "bpm_musica": 100,
                "animacao_trigger": "Thinking"
            }
        }