from pydantic import BaseModel
from typing import Any

class InterrogateRequest(BaseModel):
    session_id: str
    player_text: str

class StatusInvestigacao(BaseModel):
    nivel_suspeita: int
    congelar_input: bool
    detectou_mentira: bool
    fim_de_jogo: bool

class FeedbackVisual(BaseModel):
    cor_iluminacao: str
    bpm_musica: int
    animacao_trigger: str

class ResponseContract(BaseModel):
    id_turno: int
    texto_detetive: str
    status_investigacao: StatusInvestigacao
    feedback_visual: FeedbackVisual
