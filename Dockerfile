FROM python:3.12-slim

# ffmpeg is required by the encode stage. For the Blender render backend,
# also install Blender here (apt-get install -y blender) or bake a GPU base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SPIN360_DATA_DIR=/data \
    SPIN360_DB_URL=sqlite:////data/spin360.db \
    SPIN360_INLINE=0
VOLUME ["/data"]
EXPOSE 8000

# default command runs the API; the worker uses the same image (see compose)
CMD ["uvicorn", "spin360.api:app", "--host", "0.0.0.0", "--port", "8000"]
