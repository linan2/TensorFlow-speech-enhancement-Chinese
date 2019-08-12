train_file_dir=data/train_dir
test_file_dir=data/test_dir
val_size=1000
#####################Data prepare########################
find $train_file_dir -name '*.wav' > data/all_wav
find $test_file_dir -name '*.wav' > data/test_wav
for dataset in data/all_wav data/test_wav;do
	awk -F '/' '{print $NF}' $dataset | awk -F '.' '{print $1}' > data/wav_name
	past -d ' ' data/wav_name $dataset > data/`echo $dataset | awk -F '/' '{print $2}'`.txt
	rm data/wav_name
python scripts/get_train_val_scp.py --data_dir=$data --val_size $val_size
echo "Finish data prepare!"
date
########################################################

python2.7 pre_process_data.py calculate_train_features --train_speech_path="train.txt" --data_type=train
python2.7 pre_process_data.py calculate_train_features --train_speech_path="data/cv/inputs_dir.txt" --data_type=cv
python2.7 pre_process_test.py calculate_train_features --train_speech_path="data/test/real/rev_dir.txt" --data_type=test/real
