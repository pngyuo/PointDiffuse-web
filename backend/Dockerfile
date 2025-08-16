# 使用更轻量的基础镜像
FROM python:3.9-slim-bullseye

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# 复制必要文件
COPY requirements.txt .
COPY app.py .
COPY model_utils.py .
COPY PDiffuse_test.py .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建上传文件夹和模型占位目录
RUN mkdir -p /app/uploads && \
    mkdir -p /app/model

# 设置环境变量
ENV MODEL_PATH=/app/model/best_model.pth

CMD ["python", "app.py"]