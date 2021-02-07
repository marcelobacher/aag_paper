% remove all warnings
warning('off','all');
% add necessary paths
addpath(genpath('./afg_km'));
addpath(genpath('./ewkm'));
addpath(genpath('./enclus'));
addpath(genpath('./iforest'));
nRep = 20;
nRepVal = 10;
alpha = 0.05;
% flag to add noise (1st setting)
add_noise = 0;
noise_factors = [.01 .03 .05 .07 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
% benchmarks
benchmarks = {'aag'; 'fb'; 'enclus'; 'ewkm'; 'afg'; 'iforest'};
% datasets
datapath = '../data/';
resultspath = '../results/';
dataset_name = {...
'Features - Fourier';...
'Faults';...
'Segmentation';...
'Satimage';...
'Arrhythmia';...
'Audiology';...
'Dermatology';...
'Glass';...
'Pen Digits';...
'Features - Kar.';...
'Features - Pix';...
'Letter';...
'Zoo';...
'Wine';...
'Waveform';...
'Waveform2';...
'Isolet';...
'Sonar';...
'Breast Cancer';...
'Diabetic';...
'Lung Cancer';...
'Splice';...
'Covertype';...
'Thyroid';...
'Musk-2';...
'Arrhythmia';...
'Lympho';...
'Speech';...
'Wine';...
'Pen-Digits';...
'Mammography';...
'Cover';...
'MNIST';...
'Ionosphere';...
'BreastW';...
'Shuttle';...
'Satellite';...
'Letter';...
'Satimage';...
'Glass';...
'Thyroid';...
'KDDCup99 HTTP';...
'KDDCup99 SMTP';...
'Musk';...
'Features-Fourier';...
'Faults';...
'Waveform';...
'Waveform2';...
'Features-Kar';...
'Features-Pix';...
'Dermatology';...
'Isolet';...
'Sonar';...
'Diabetic';...
};

dataset_file = {...
'data_fou.mat';...
'faults_clean.mat';...
'segmentation_clean.mat';...
'satimages_all.mat';...
'arrhythmia_clean.mat';...
'audiology_clean.mat';...
'dermatology_clean.mat';...
'glass_clean.mat';...
'pendigits_clean.mat';...
'data_kar.mat';...
'data_pix.mat';...
'letter_clean.mat';...
'zoo_clean.mat';...
'wine_clean.mat';...
'waveform_clean.mat';...
'waveform2_clean.mat';...
'isolet_clean.mat';...
'sonar_clean.mat';...
'breast-cancer-wisconsin_clean.mat';...
'diabetic_clean.mat';...
'lung-cancer_clean.mat';...
'splice_clean.mat';...
'covertype_clean.mat';...
'thyroid_clean.mat';...
'musk_odds.mat';...
'arrhythmia_odds.mat';...
'lympho_odds.mat';...
'speech_odds.mat';...
'wine_odds.mat';...
'pendigits_odds.mat';...
'mammography_odds.mat';...
'cover_odds.mat';...
'mnist_odds.mat';...
'ionosphere_odds.mat';...
'breastw_odds.mat';...
'shuttle_odds.mat';...
'satellite_odds.mat';...
'letter_odds.mat';...
'satimage-2_odds.mat';...
'glass_odds.mat';...
'thyroid_odds.mat';...
'http_odds.mat';...
'smtp_odds.mat';...
'musk_odds.mat';...
'Features-Fourier_odds.mat';...
'faults_odds.mat';...
'waveform_odds.mat';...
'waveform2_odds.mat';...
'Features-Kar_odds.mat';...
'Features-Pix_odds.mat';...
'dermatology_odds.mat';...
'isolet_odds.mat';...
'sonar_odds.mat';...
'diabetic_odds.mat';...
};
nFiles = length(dataset_file);
for i=1:nFiles
  fprintf('### Working on file: %s (%d/%d)...\n', ...
    [datapath dataset_file{i}], i, nFiles);
  try
    f = load([datapath dataset_file{i}]);
    if isfield(f, 'data') && isfield(f, 'labels')
      data = f.data;
      labels = f.labels;
    elseif isfield(f, 'X') && isfield(f, 'y')
      data = f.X;
      labels = f.y;
    else
      error('### ERROR!! Format of %s is unknown\n!', dataset_file{i});
    end
    performance = run_aag_paper(data, labels, benchmarks, alpha, ...
      add_noise, noise_factors, nRep, nRepVal);
    ymdxxx = clock;
    if ymdxxx(2) < 10
      month = ['0' num2str(ymdxxx(2))];
    else
      month = num2str(ymdxxx(2));
    end
    if ymdxxx(3) < 10
      day = ['0' num2str(ymdxxx(3))];
    else
      day = num2str(ymdxxx(3));
    end
    targetfile = [resultspath sprintf('%d%s%s_results_%s', ...
      ymdxxx(1), month, day, dataset_file{i})];    
    fprintf('### Recording file: %s...\n', targetfile);
    save(targetfile, 'performance');
  catch e
    disp(e)
  end
end
warning('on','all');