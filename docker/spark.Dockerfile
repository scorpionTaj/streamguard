# Spark 4.0 + the Python deps the feature job needs (pandas/pyarrow for
# applyInPandasWithState, redis, pyyaml). The repo is mounted at /opt/app.
FROM apache/spark:4.0.0
USER root
RUN pip3 install --no-cache-dir "numpy<2.3" "pandas>=2.2,<3" "pyarrow>=16,<20" "redis>=5" "pyyaml>=6"
USER spark
ENV PYTHONPATH=/opt/app/src
ENV PYTHONDONTWRITEBYTECODE=1
