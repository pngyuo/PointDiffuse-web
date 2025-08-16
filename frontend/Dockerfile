# 前端Dockerfile (支持环境变量注入)
FROM node:18 as builder

WORKDIR /app

# 1. 设置构建参数
ARG REACT_APP_API_BASE_URL
ENV REACT_APP_API_BASE_URL=$REACT_APP_API_BASE_URL

# 2. 首先复制包管理文件
COPY package.json package-lock.json ./

# 3. 清理npm缓存并安装依赖
RUN npm cache clean --force && \
    npm install --legacy-peer-deps --prefer-offline --no-audit --no-fund

# 4. 复制所有其他文件
COPY . .

# 5. 验证项目结构
RUN ls -la /app/public && ls -la /app/src

# 6. 构建应用
RUN npm run build

# 生产阶段
FROM nginx:alpine

# 复制构建产物
COPY --from=builder /app/build /usr/share/nginx/html

# 复制自定义nginx配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]