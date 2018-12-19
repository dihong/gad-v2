# Insider Threat Detection (final)

The final version of the insider threat detection. It has the following features:
  - Extracts more compact and discriminative features.
  - Propose graph based detection algorithm to improve performance.

## Requirements
    - Apache Spark
    - pip install wrapt
    - pip install pgmpy

## Prepare data

- Download CERT data [r6.2.tar.bz2](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)

- Download [answers.tar.bz2](ftp://ftp.sei.cmu.edu/pub/cert-data/answers.tar.bz2)

- Extract both r6.2.tar.bz2 and answers.tar.bz2, and place extracted answers under r6.2 folder.

## Important Configurations: edit src/config.py
    - SPARK_MASTER: master address of the Spark.
    - config.io.data_dir: root of the extracted r6.2 data.
## Run the codes from src directory
    - bash run.sh
## Outputs
    - cache: all necessary intermediate results.
    - result: scores of baseline systems.
    - CR scores: printed to the terminal with highted colors.
### Results

Table 1. The Cumulative Recall (CR) for 400

|         Algorithms         |   PCA            | SVM | ISO-Forest | DNN |
| --------------------- | ---------------- | ----- | ---------- | ---- |
| No GTM | 13.64 | 10.36  |     8.10     | 13.91   |
| GTM Enabled | 15.00 | 12.00  |     11.27      | 15.54   |

Table 2. The Cumulative Recall (CR) for 1000

|         Algorithms         |   PCA            | SVM | ISO-Forest | DNN |
| --------------------- | ---------------- | ----- | ---------- | ---- |
| No GTM | 37.18 | 34.36  |     32.10     | 36.45   |
| GTM Enabled | 39.00 | 35.73  |     35.27      | 39.54   |


