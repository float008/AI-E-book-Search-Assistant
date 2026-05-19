# 使用 Elastic 官方镜像，勿用本地 elasticsearch:8.17.0（可能已被旧构建打上 IK 标签）
FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.0

# 安装 IK 分词（版本须与 ES 一致）
RUN rm -rf /usr/share/elasticsearch/plugins/analysis-ik && \
    elasticsearch-plugin install --batch \
      https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-8.17.0.zip
