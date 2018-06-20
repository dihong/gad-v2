# Insider Threat Detection v2

The second version (v2) of the insider threat detection. The v2 has the following features:
 - Develops user specific models for anomaly detection.
  - Extracts more compact and discriminative features.
   - Employs naive bayes based probablistic models with Bernoulli or Poisson distributions.

## Requirements
    - Apache Spark
## Run the codes
     - Extract features
     ```sh
     $ cd extra-features:
     $ /spark/bin/spark-submit --driver-memory 200g extract_user_email_and_pc.py
     $ /spark/bin/spark-submit --driver-memory 120g extract_relational_features.py
     ```
      - Train & test naive bayes models:
      ```sh
      $ cd src:
      $ /spark/bin/spark-submit --driver-memory 120g train_naive_bayes.py
      $ /spark/bin/spark-submit --driver-memory 120g test_naive_bayes.py
      ```
