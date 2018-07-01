# /spark/bin/spark-submit --driver-memory 120g train_iso_forest.py
# /spark/bin/spark-submit --driver-memory 120g train_naive_bayes.py
# /spark/bin/spark-submit --driver-memory 120g test_naive_bayes.py
# /spark/bin/spark-submit --driver-memory 120g iso_forest_by_users.py
# /spark/bin/spark-submit --driver-memory 120g iso_forest_by_days.py
# /spark/bin/spark-submit --driver-memory 120g train_naive_bayes.py && /spark/bin/spark-submit --driver-memory 120g test_naive_bayes.py
python run_dnn.py |& tee dnn.log
