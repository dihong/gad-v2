# Insider Threat Detection (final)

The final version of the insider threat detection. It has the following features:
  - Extracts more compact and discriminative features.
  - Propose graph based detection algorithm to improve performance.

## Requirements
    - Apache Spark
    - pip install wrapt
    - pip install pgmpy
## Prepare data
    - Download CERT data [r6.2.tar.bz2](ftp://ftp.sei.cmu.edu/pub/cert-data/r6.2.tar.bz2).
    - Download [answers.tar.bz2](ftp://ftp.sei.cmu.edu/pub/cert-data/answers.tar.bz2).
    - Extract both r6.2.tar.bz2 and answers.tar.bz2, and place extracted answers under r6.2 folder.
## Important Configurations: edit src/config.py
    - SPARK_MASTER: master address of the Spark.
    - config.io.data_dir: root of the extracted r6.2 data.
## Run the codes from the src directory
    - bash run.sh
## Outputs
    - cache: all necessary intermediate results.
    - result: scores of baseline systems.
    - CR scores: printed to the terminal with highted colors.
