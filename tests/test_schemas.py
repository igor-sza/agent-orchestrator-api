from app.schemas import InterrogateRequest, ResponseContract, StatusInvestigacao, FeedbackVisual


def test_interrogate_request_validation():
    req = InterrogateRequest(session_id="sess1", player_text="Eu não sei.")
    assert req.session_id == "sess1"
    assert isinstance(req.player_text, str)


def test_response_contract_validation():
    status = StatusInvestigacao(nivel_suspeita=10, congelar_input=False, detectou_mentira=False, fim_de_jogo=False)
    feedback = FeedbackVisual(cor_iluminacao="#FFFFFF", bpm_musica=90, animacao_trigger="Neutral")
    resp = ResponseContract(id_turno=1, texto_detetive="Teste", status_investigacao=status, feedback_visual=feedback)
    assert resp.id_turno == 1
    assert resp.status_investigacao.nivel_suspeita == 10
