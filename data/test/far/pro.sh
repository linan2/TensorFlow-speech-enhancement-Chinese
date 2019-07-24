find /Work18/2017/linan/SE/my_enh/cut_data/SIMU/far_evl/ -name *.wav > far_dir
paste -d ' ' far_dir cln_et.txt > inputs_dir.txt
paste -d   name inputs_feat.txt > inputs.txt
awk -F / {print } inputs_feat.txt | awk -F . {print } > name
find /Work18/2017/linan/SE/my_enh/workspace/features/spectrogram/test/far/ -name *.p > inputs_feat.txt
