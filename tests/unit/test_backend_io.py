from types import SimpleNamespace

from spin_rag.spin_rag import BACKEND_LLAMACPP, _Backend


class _FakeCompletions:
    def __init__(self, content="hello", empty=False):
        self.content = content
        self.empty = empty
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.empty:
            return SimpleNamespace(choices=[])
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeEmbeddings:
    def __init__(self, error=None):
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2]),
                SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )


def _backend_with_client(chat=None, embed=None):
    backend = _Backend(backend=BACKEND_LLAMACPP)
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=chat or _FakeCompletions()),
        embeddings=embed or _FakeEmbeddings(),
    )
    return backend


def test_chat_returns_content():
    backend = _backend_with_client(chat=_FakeCompletions("ok"))
    assert backend.chat("llama", [{"role": "user", "content": "hi"}], max_tokens=8) == "ok"


def test_chat_empty_choices():
    backend = _backend_with_client(chat=_FakeCompletions(empty=True))
    assert backend.chat("llama", [{"role": "user", "content": "hi"}]) == ""


def test_embed_filters_empty_and_unwraps():
    backend = _backend_with_client()
    assert backend.embed("llama", "") == []
    vecs = backend.embed("llama", ["a", "b"])
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
