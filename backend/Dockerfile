# 支持真实 PointDiffuse 模型的 Dockerfile
FROM python:3.9-slim-bullseye

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# 创建完整的requirements.txt（包含真实模型需要的依赖）
RUN echo "torch==1.13.0" > requirements.txt && \
    echo "torchvision==0.14.0" >> requirements.txt && \
    echo "numpy==1.24.3" >> requirements.txt && \
    echo "matplotlib==3.7.1" >> requirements.txt && \
    echo "Pillow==9.5.0" >> requirements.txt && \
    echo "flask==2.3.2" >> requirements.txt && \
    echo "flask-cors==4.0.0" >> requirements.txt && \
    echo "werkzeug==2.3.6" >> requirements.txt && \
    echo "scikit-learn==1.3.0" >> requirements.txt && \
    echo "scipy==1.10.1" >> requirements.txt && \
    echo "tqdm==4.65.0" >> requirements.txt && \
    echo "seaborn==0.12.2" >> requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY model_utils.py .

# 复制 PointDiffuse 相关代码（如果存在）
COPY PDiffuse_train.py* ./
COPY PDiffuse_test.py* ./

# 创建必要的目录
RUN mkdir -p /app/uploads && \
    mkdir -p /app/model

# 设置环境变量
ENV MODEL_PATH=/app/model/best_model.pth
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]