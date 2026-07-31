FROM python:3.12-slim

# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1: OpenCV runtime deps needed by
# paddleocr/paddlepaddle. libgomp1: OpenMP runtime needed by paddlepaddle.
# libicu76/libgraphite2-3/libharfbuzz0b/libfontconfig1: shared libs tectonic
# dynamically links against at runtime (text shaping/font config).
# curl/ca-certificates: fetch the tectonic binary below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libicu76 \
    libgraphite2-3 \
    libharfbuzz0b \
    libfontconfig1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# tectonic (LaTeX engine used by services/template_service.py) is not on PyPI;
# install the single-binary release and put it on PATH.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/tectonic

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads data/rendered logs
