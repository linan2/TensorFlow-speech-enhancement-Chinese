stage=0
if [ $stage -le 0 ]; then
python evaluate.py calculate_pesq --workspace='workspace' --speech_dir='cut_data/SIMU/cln_evl' --type='test2/far'

cat _pesq_results.txt|tail -n 539 |head -n 538|awk '{sum+=$2} END {print "far_pesq = ", sum/NR}' > avr_pesq
wait
fi
if [ $stage -le 1 ]; then
python evaluate.py calculate_pesq --workspace='workspace' --speech_dir='cut_data/SIMU/cln_evl' --type='test2/near'

cat _pesq_results.txt|tail -n 539 |head -n 538|awk '{sum+=$2} END {print "near_pesq = ", sum/NR}' >> avr_pesq
wait
fi

if [ $stage -le 2 ]; then
python evaluate.py calculate_pesq --workspace='workspace' --speech_dir='cut_data/Real/cln_et' --type='test2/real'

cat _pesq_results.txt|tail -n 373 |head -n 372|awk '{sum+=$2} END {print "real_pesq = ", sum/NR}' >> avr_pesq
wait
fi
cat avr_pesq
