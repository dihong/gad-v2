#/spark/bin/spark-submit --driver-memory 120g extract_compact_features.py && \
#/spark/bin/spark-submit --driver-memory 120g baselines.py && \
/spark/bin/spark-submit --driver-memory 120g extract_bayesian_features.py && \
/spark/bin/spark-submit --driver-memory 120g bayesian_network.py
