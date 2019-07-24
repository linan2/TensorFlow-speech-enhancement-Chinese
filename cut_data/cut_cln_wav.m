% ----------------------------------------------------------------------------------------------------
% parameters and configures
% ----------------------------------------------------------------------------------------------------
%dir_name = {'c31/','c34/','c35/','c38/','c3c/','c3d/','c3f/','c3j/','c3k/','c3l/','c3p/','c3s/','c3t/','c3w/','c3z/','c40/','c41/','c42/','c45/','c49/'};
%dir_name = {'c30/','c32/','c33/','c37/','c39/','c3b/','c3h/','c3o/','c3q/','c3r/','c3y/','c46/','c48/','c4a/'};
%dir_name = {'c36/','c3a/','c3e/','c3g/','c3i/','c3m/','c3n/','c3u/','c3v/','c3x/','c43/','c44/','c47/','c4b/'};
%dir_name = {'c02/','c05/','c08/','c0b/','c0e/','c0h/','c0k/','c0n/','c0q/','c0t/','c0w/','c0z/','c12/','c15/','c18/','c1b/','c1e/','c1h/','c1k/','c1n/','c1q/','c1t/','c1w/','c1z/','c22/','c25/','c28/','c2b/','c2e/','c2h/','c2k/','c03/','c06/','c09/','c0c/','c0f/','c0i/','c0l/','c0o/','c0r/','c0u/','c0x/','c10/','c13/','c16/','c19/','c1c/','c1f/','c1i/','c1l/','c1o/','c1r/','c1u/','c1x/','c20/','c23/','c26/','c29/','c2c/','c2f/','c2i/','c2l/','c04/','c07/','c0a/','c0d/','c0g/','c0j/','c0m/','c0p/','c0s/','c0v/','c0y/','c11/','c14/','c17/','c1a/','c1d/','c1g/','c1j/','c1m/','c1p/','c1s/','c1v/','c1y/','c21/','c24/','c27/','c2a/','c2d/','c2g/','c2j/'};
dir_name ={'c02/'};
%disp(length(dir_name));

% ---------------------------------------------------------------------------------------------------
% cut wavforms
% --------------------------------------------------------------------------------------------------
for t=1:length(dir_name)
    % get the current sub-directory
    tempdir=dir_name{t};
    disp(tempdir);
    % define the path of reverberation wavforms and enhanced wavforms
    clean_filedir = ['/CDShare/REVERB_DATA/raw_wsj0_data/data/primary_microphone/si_tr/',tempdir];
    enh_filedir = ['/Work18/2015/gemeng/se/mydnn/tools/MSLP/MCMSLP_L750_D512/dereverb_GSSn1a1b0.15/si_tr/',tempdir,'/1/RAW/'];
    % get all the file names of enhanced wavforms
    dirOutput = dir([enh_filedir, '*_2.wav']);
    file_name = {dirOutput.name}';
    disp(file_name);
    [rows,cols] = size(file_name); 
    
    % cut the reverberation wavforms based on the length of the corresponding enhanced wavforms
    save_path = ['/Work18/2015/gemeng/se/mydnn/tools/MSLP/MCMSLP_L750_D512/dereverb_GSSn1a1b0.15/cln_cut/si_tr/',tempdir];
    mkdir(save_path);
    for i=1:rows
        enh_na = file_name{i};
        clean_na = [enh_na(1:8),'.wav'];
        disp(clean_na)
        %na = file_name(i);
        %audiopath=dir([filedir,file_name{i}]);
        [clean_x, Fs] = audioread([clean_filedir, clean_na]);
        [enh_x,Fs] = audioread([enh_filedir, enh_na]);
        %[r,c]=size(x);
        %if c > 1
        %    disp(na);
        %end;
        y = clean_x(1:length(enh_x));
        wrt_path = [save_path, clean_na];
        audiowrite(wrt_path,y,Fs);
    end;
    %disp(x);
    %disp(Fs);
end;
