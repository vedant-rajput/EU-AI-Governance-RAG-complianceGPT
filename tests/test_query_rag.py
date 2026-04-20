from scripts.query_rag import _wrap, detect_question_type


def test_detect_question_type():
    assert detect_question_type("What is the fine under GDPR?") == "factual"
    assert detect_question_type("Are there any exemptions in the EU AI Act?") == "yes_no"
    assert detect_question_type("Compare GDPR and NIST AI RMF") == "comparison"
    assert detect_question_type("List the main principles of GDPR") == "listing"
    assert detect_question_type("Provide a general overview of the framework") == "open_ended"

def test_wrap_text():
    text = "This is a very long text that needs to be wrapped properly."
    wrapped = _wrap(text, 10)
    assert wrapped[0] == "This is a "
    assert wrapped[1] == "very long "
    assert len(wrapped) > 2
