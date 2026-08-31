FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git universal-ctags \
    && rm -rf /var/lib/apt/lists/*

# The checkouts are bind-mounted and owned by the host user, so git would refuse them.
RUN git config --system --add safe.directory '*'

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt pytest

ENV HF_HOME=/opt/hf \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Baked in so a run needs no network: otherwise the model download fails and the token
# count falls back to a word count, which silently changes what the budget means.
RUN python -c "from sentence_transformers import SentenceTransformer; \
        SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX /opt/hf /opt/tiktoken

CMD ["bash"]
