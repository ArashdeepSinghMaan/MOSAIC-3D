FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MOSAIC-3D
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["api"]
