python2.7 pre_process_data.py calculate_train_features --train_speech_path="train.txt" --data_type=train
python2.7 pre_process_data.py calculate_train_features --train_speech_path="data/cv/inputs_dir.txt" --data_type=cv
python2.7 pre_process_test.py calculate_train_features --train_speech_path="data/test/real/rev_dir.txt" --data_type=test/real
python2.7 pre_process_test.py calculate_train_features --train_speech_path="data/test/far/far_dir.txt" --data_type=test/far
python2.7 pre_process_test.py calculate_train_features --train_speech_path="data/test/near/near_dir.txt" --data_type=test/near
