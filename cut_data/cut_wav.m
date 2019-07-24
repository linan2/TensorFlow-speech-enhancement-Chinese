filename1 = 'si_tr.txt'
[name1,path1] = textread(filename1,'%s %s')
filename2 = 'REVERB_WSJCAM0_tr.txt'
[name2,path2] = textread(filename2,'%s %s')
wavlist1 = path1;
wavlist1 = [wavlist1];
wavlist2 = path2;
wavlist2 = [wavlist2];


for i=1:7861
     wav_cln = audioread(wavlist1{i});
     wav_rev = audioread(wavlist2{i});
     disp(wavlist1{i});
     disp(wavlist2{i});
     Fs = 16000
     disp(length(wav_cln));
     disp(length(wav_rev))
     y = wav_rev(1:length(wav_cln));
     str1 = '.wav'
     wrt_path = ['reverb/',name2{i},str1];
     audiowrite(wrt_path,y,Fs);
end;
