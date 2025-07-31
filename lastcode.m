%% ======================== MAIN PROCESSING PIPELINE ========================
% 1. Initialization
data_dir = 'D:\dsp project\Data';
export_dir = 'D:\DSP Project\Data';
target_Fs = 4000; % Target sampling rate

% 2. Data Loading
[all_heart_data, all_Fs_heart, all_heart_filenames] = load_audio_files(fullfile(data_dir, 'HeartSounds\physionet'), 'wav');
[all_lung_data, all_Fs_lung, all_lung_filenames] = load_audio_files(fullfile(data_dir, 'LungSounds'), 'wav');

% 3. Preprocessing
[processed_heart_data, Fs_heart_processed] = resample_data(all_heart_data, all_Fs_heart, target_Fs);
[processed_lung_data, Fs_lung_processed] = resample_data(all_lung_data, all_Fs_lung, target_Fs);

[filtered_heart_data, b_h, a_h] = filter_data(processed_heart_data, Fs_heart_processed, [20 1000]);
[filtered_lung_data, b_l, a_l] = filter_data(processed_lung_data, Fs_lung_processed, [100 1999]);

normalized_heart_data = normalize_data(filtered_heart_data);
normalized_lung_data = normalize_data(filtered_lung_data);

% 4. Feature Extraction
[mfccs_heart, mfccs_lung] = extract_mfccs(normalized_heart_data, normalized_lung_data, target_Fs);
[heart_features_summary, lung_features_summary] = extract_summary_features(normalized_heart_data, normalized_lung_data, target_Fs);

% 5. Heart Sound Processing
[all_heart_segment_features, Y_heart_combined] = process_heart_sounds(normalized_heart_data, all_heart_filenames, Fs_heart_processed, data_dir);

% 6. Lung Sound Processing  
[all_lung_segment_features, Y_lung_combined] = process_lung_sounds(normalized_lung_data, all_lung_filenames, Fs_lung_processed, data_dir);

% 7. Data Consolidation
[X_heart_combined, Y_heart_combined, all_heart_feature_names] = consolidate_heart_features(all_heart_segment_features, Y_heart_combined);
[X_lung_combined, Y_lung_combined, lung_feature_names] = consolidate_lung_features(all_lung_segment_features, Y_lung_combined);

% 8. Export Data
export_features(X_heart_combined, Y_heart_combined, X_lung_combined, Y_lung_combined, export_dir);

% 9. Visualization (All plotting at the end)
create_visualizations(normalized_heart_data, normalized_lung_data, ...
                     mfccs_heart, mfccs_lung, ...
                     X_heart_combined, Y_heart_combined, ...
                     X_lung_combined, Y_lung_combined, ...
                     all_heart_feature_names, lung_feature_names, ...
                     Fs_heart_processed, Fs_lung_processed, ...
                     all_heart_filenames, export_dir);

%% ======================== HELPER FUNCTIONS ========================
function [audio_data, sample_rates, filenames] = load_audio_files(folder, ext)
    files = dir(fullfile(folder, ['*.', ext]));
    audio_data = cell(length(files), 1);
    sample_rates = zeros(length(files), 1);
    filenames = cell(length(files), 1);
    
    for i = 1:length(files)
        [audio_data{i}, sample_rates(i)] = audioread(fullfile(folder, files(i).name));
        filenames{i} = files(i).name;
    end
end

function [processed_data, Fs_processed] = resample_data(data, original_Fs, target_Fs)
    processed_data = cell(size(data));
    for i = 1:length(data)
        if original_Fs(i) ~= target_Fs
            processed_data{i} = resample(data{i}, target_Fs, original_Fs(i));
        else
            processed_data{i} = data{i};
        end
    end
    Fs_processed = target_Fs;
end

function [filtered_data, b, a] = filter_data(data, Fs, freq_range)
    % Verify frequencies are within valid range
    nyquist = Fs/2;
    if any(freq_range <= 0) || any(freq_range >= nyquist)
        error('Frequencies must be between 0 and %f Hz (Nyquist frequency)', nyquist);
    end
    
    % Normalize frequencies
    Wn = freq_range/nyquist;
    
    % Design filter
    [b, a] = butter(4, Wn, 'bandpass');
    
    % Apply filter
    filtered_data = cell(size(data));
    for i = 1:length(data)
        filtered_data{i} = filtfilt(b, a, data{i});
    end
end

function normalized_data = normalize_data(data)
    normalized_data = cell(size(data));
    for i = 1:length(data)
        if ~isempty(data{i}) && max(abs(data{i})) > 0
            normalized_data{i} = data{i} / max(abs(data{i}));
        else
            normalized_data{i} = data{i};
        end
    end
end

function [mfccs_heart, mfccs_lung] = extract_mfccs(heart_data, lung_data, Fs)
    % MFCC parameters
    window_length_mfcc = 0.025; % 25 ms window
    overlap_length_mfcc = 0.01;  % 10 ms overlap
    frame_len_samples = round(window_length_mfcc * Fs);
    hop_len_samples = round(overlap_length_mfcc * Fs);
    num_mfccs_desired = 13;

    % Initialize cell arrays
    mfccs_heart = cell(size(heart_data));
    mfccs_lung = cell(size(lung_data));

    % Heart Sound MFCCs
    for i = 1:length(heart_data)
        current_data = heart_data{i};
        if ~isempty(current_data)
            if size(current_data, 2) > size(current_data, 1)
                current_data = current_data';
            end
            
            afe = audioFeatureExtractor(...
                'SampleRate', Fs, ...
                'Window', hann(frame_len_samples, 'periodic'), ...
                'OverlapLength', hop_len_samples, ...
                'mfcc', true);
            
            current_mfccs = extract(afe, current_data);
            
            if size(current_mfccs, 2) >= num_mfccs_desired
                current_mfccs = current_mfccs(:, 1:num_mfccs_desired);
            else
                warning('Not enough MFCCs extracted for heart sound %d', i);
                current_mfccs = [current_mfccs, zeros(size(current_mfccs,1), num_mfccs_desired-size(current_mfccs,2))];
            end
            mfccs_heart{i} = current_mfccs;
        else
            mfccs_heart{i} = [];
        end
    end

    % Lung Sound MFCCs
    for i = 1:length(lung_data)
        current_data = lung_data{i};
        if ~isempty(current_data)
            if size(current_data, 2) > size(current_data, 1)
                current_data = current_data';
            end
            
            afe = audioFeatureExtractor(...
                'SampleRate', Fs, ...
                'Window', hann(frame_len_samples, 'periodic'), ...
                'OverlapLength', hop_len_samples, ...
                'mfcc', true);
            
            current_mfccs = extract(afe, current_data);
            
            if size(current_mfccs, 2) >= num_mfccs_desired
                current_mfccs = current_mfccs(:, 1:num_mfccs_desired);
            else
                warning('Not enough MFCCs extracted for lung sound %d', i);
                current_mfccs = [current_mfccs, zeros(size(current_mfccs,1), num_mfccs_desired-size(current_mfccs,2))];
            end
            mfccs_lung{i} = current_mfccs;
        else
            mfccs_lung{i} = [];
        end
    end
end


function [heart_features, lung_features] = extract_summary_features(heart_data, lung_data, Fs)
    % Initialize structures
    heart_features = struct('filename', {}, 'RMS_Mean', {}, 'ZCR_Mean', {}, 'SpecCentroid_Mean', {});
    lung_features = struct('filename', {}, 'RMS_Mean', {}, 'ZCR_Mean', {}, 'SpecCentroid_Mean', {});

    % ========== Heart Sound Processing ==========
    for i = 1:length(heart_data)
        current_data = heart_data{i};
        if isempty(current_data) || all(current_data == 0)
            heart_features(i).RMS_Mean = NaN;
            heart_features(i).ZCR_Mean = NaN;
            heart_features(i).SpecCentroid_Mean = NaN;
            continue;
        end
        
        % Ensure column vector
        current_data = current_data(:);
        
        % RMS
        rms_val = rms(current_data);
        
        % Zero Crossing Rate
        zcr_val = sum(abs(diff(sign(current_data)))) / (2 * length(current_data));
        
        % Spectral Centroid
        N = length(current_data);
        Y = fft(current_data);
        P2 = abs(Y/N);
        half_len = floor(N/2)+1;  % Integer division
        P1 = P2(1:half_len);
        P1(2:end-1) = 2*P1(2:end-1);
        freq_axis = (0:half_len-1) * (Fs/N);
        
        if sum(P1) > 0
            spectral_centroid = sum(freq_axis(:) .* P1(:)) / sum(P1);
        else
            spectral_centroid = 0;
        end
        
        % Store features
        heart_features(i).RMS_Mean = rms_val;
        heart_features(i).ZCR_Mean = zcr_val;
        heart_features(i).SpecCentroid_Mean = spectral_centroid;
    end

    % ========== Lung Sound Processing ==========
    for i = 1:length(lung_data)
        current_data = lung_data{i};
        if isempty(current_data) || all(current_data == 0)
            lung_features(i).RMS_Mean = NaN;
            lung_features(i).ZCR_Mean = NaN;
            lung_features(i).SpecCentroid_Mean = NaN;
            continue;
        end
        
        % Ensure column vector
        current_data = current_data(:);
        
        % RMS Energy
        rms_val = rms(current_data);
        
        % Zero Crossing Rate
        zcr_val = sum(abs(diff(sign(current_data)))) / (2 * length(current_data));
        
        % Spectral Centroid
        N = length(current_data);
        Y = fft(current_data);
        P2 = abs(Y/N);
        half_len = floor(N/2)+1;  % Integer division
        P1 = P2(1:half_len);
        P1(2:end-1) = 2*P1(2:end-1);
        freq_axis = (0:half_len-1) * (Fs/N);
        
        if sum(P1) > 0
            spectral_centroid = sum(freq_axis(:) .* P1(:)) / sum(P1);
        else
            spectral_centroid = 0;
        end
        
        % Store features
        lung_features(i).RMS_Mean = rms_val;
        lung_features(i).ZCR_Mean = zcr_val;
        lung_features(i).SpecCentroid_Mean = spectral_centroid;
    end
end
function [S1_locs, S2_locs, systole_intervals, diastole_intervals] = segmentHeartSounds(heart_audio, Fs)
    if isempty(heart_audio) || all(heart_audio == 0)
        S1_locs = []; S2_locs = []; systole_intervals = []; diastole_intervals = [];
        return;
    end

    % 1. Preprocessing
    abs_audio = abs(heart_audio);
    cutoff_freq_envelope = 10; % Hz
    [b_lp, a_lp] = butter(4, cutoff_freq_envelope/(Fs/2), 'low');
    envelope = filtfilt(b_lp, a_lp, abs_audio);
    envelope = envelope / max(envelope);

    % 2. Peak Detection
    min_peak_dist_samples = round(0.2 * Fs);
    [~, locs] = findpeaks(envelope, 'MinPeakProminence', 0.2, 'MinPeakDistance', min_peak_dist_samples);
    
    if isempty(locs) || length(locs) < 2
        S1_locs = []; S2_locs = []; systole_intervals = []; diastole_intervals = [];
        return;
    end

    % 3. S1/S2 Classification
    candidate_intervals = diff(locs);
    median_interval = median(candidate_intervals);
    
    S1_locs = [];
    S2_locs = [];
    
    for k = 1:length(locs)-1
        if candidate_intervals(k) < median_interval * 0.8
            S1_locs = [S1_locs, locs(k)];
            S2_locs = [S2_locs, locs(k+1)];
        end
    end
    
    % 4. Interval Calculation
    systole_intervals = [];
    diastole_intervals = [];
    
    for k = 1:min(length(S1_locs), length(S2_locs))
        if S2_locs(k) > S1_locs(k)
            systole_intervals = [systole_intervals; S1_locs(k), S2_locs(k)];
            
            if k < length(S1_locs) && S1_locs(k+1) > S2_locs(k)
                diastole_intervals = [diastole_intervals; S2_locs(k), S1_locs(k+1)];
            end
        end
    end
    
    % Filter invalid segments
    if ~isempty(systole_intervals)
        systole_intervals = systole_intervals(systole_intervals(:,2) > systole_intervals(:,1), :);
    end
    
    if ~isempty(diastole_intervals)
        diastole_intervals = diastole_intervals(diastole_intervals(:,2) > diastole_intervals(:,1), :);
    end
    
    % Ensure uniqueness and sorting
    S1_locs = unique(S1_locs);
    S2_locs = unique(S2_locs);
    S1_locs = sort(S1_locs);
    S2_locs = sort(S2_locs);
end
function [segment_features, Y_heart_combined] = process_heart_sounds(data, filenames, Fs, data_dir)
    % Initialize outputs
    segment_features = cell(size(data));
    Y_heart_combined = cell(length(filenames), 1);
    
    % Load annotations
    heart_annotations_path = fullfile(data_dir, 'Online Appendix_training set.csv');
    if exist(heart_annotations_path, 'file')
        heart_annotations = readtable(heart_annotations_path, 'VariableNamingRule', 'preserve');
        
        % Verify we have the expected columns
        if ~all(ismember({'Challenge record name', 'Class (-1=normal 1=abnormal)'}, ...
                         heart_annotations.Properties.VariableNames))
            error('Annotation file missing required columns. Found: %s', ...
                  strjoin(heart_annotations.Properties.VariableNames, ', '));
        end
    else
        heart_annotations = [];
        warning('Annotations not found at: %s', heart_annotations_path);
    end

    for i = 1:length(data)
        current_data = data{i};
        current_filename = filenames{i};
        
        if isempty(current_data) || all(current_data == 0)
            segment_features{i} = struct();
            Y_heart_combined{i} = 'UNLABELED';
            continue;
        end

        % Segmentation
        [S1_locs, S2_locs, systole_intervals, diastole_intervals] = segmentHeartSounds(current_data, Fs);
        
        % Feature extraction per segment
        file_segment_features = [];
        
        % Systolic segments
        for j = 1:size(systole_intervals, 1)
            segment = current_data(systole_intervals(j,1):systole_intervals(j,2));
            features = extractHeartSegmentFeatures(segment, Fs);
            features.SegmentType = 'systole';
            file_segment_features = [file_segment_features; features];
        end
        
        % Diastolic segments
        for j = 1:size(diastole_intervals, 1)
            segment = current_data(diastole_intervals(j,1):diastole_intervals(j,2));
            features = extractHeartSegmentFeatures(segment, Fs);
            features.SegmentType = 'diastole';
            file_segment_features = [file_segment_features; features];
        end
        
        segment_features{i} = file_segment_features;
        
        % Label assignment
        if ~isempty(heart_annotations)
            [~, record_name, ~] = fileparts(current_filename);
            
            % Find matching record - try both challenge and original record names
            idx = find(strcmpi(heart_annotations.('Challenge record name'), record_name) | ...
                       strcmpi(heart_annotations.('Original record name'), record_name), 1);
            
            if ~isempty(idx)
                raw_label = heart_annotations.('Class (-1=normal 1=abnormal)')(idx);
                
                if raw_label == -1
                    Y_heart_combined{i} = 'normal';
                elseif raw_label == 1
                    Y_heart_combined{i} = 'abnormal';
                else
                    Y_heart_combined{i} = 'unknown';
                    warning('Unexpected label value %d for file %s', raw_label, current_filename);
                end
            else
                Y_heart_combined{i} = 'unmatched';
                warning('No annotation found for record: %s', record_name);
            end
        else
            Y_heart_combined{i} = 'unmatched_no_table';
        end
    end
end

function [segment_features, Y_lung_combined] = process_lung_sounds(data, filenames, Fs, data_dir)
    % Initialize
    segment_features = cell(size(data));
    Y_lung_combined = cell(length(filenames), 1);
    
    % Load annotations
    lung_annotations_path = fullfile(data_dir, 'lung_annotations.csv');
    if exist(lung_annotations_path, 'file')
        lung_annotations = readtable(lung_annotations_path, 'VariableNamingRule', 'preserve');
    else
        error('Lung annotations file not found at %s', lung_annotations_path);
    end

    for i = 1:length(data)
        current_data = data{i};
        current_filename = filenames{i};
        
        if isempty(current_data) || all(current_data == 0)
            segment_features{i} = struct();
            Y_lung_combined{i} = 'UNLABELED';
            continue;
        end

        % Extract features (now using the passed Fs parameter)
        features = extractLungSoundFeatures(current_data, Fs);
        segment_features{i} = features;
        
        % Label assignment
        idx = find(strcmp(lung_annotations.filename, current_filename), 1);
        if ~isempty(idx)
            Y_lung_combined{i} = lung_annotations.quality{idx};
        else
            Y_lung_combined{i} = 'UNLABELED_NO_MATCH';
            warning('Lung sound %s has no match in annotations', current_filename);
        end
    end
end
function [X, Y, feature_names] = consolidate_heart_features(segment_features, labels)
     % Get all feature names
    all_feature_names = {};
    for i = 1:length(segment_features)
        if ~isempty(segment_features{i})
            all_feature_names = union(all_feature_names, fieldnames(segment_features{i}));
        end
    end
    all_feature_names = setdiff(all_feature_names, {'SegmentType'});
    
    % Initialize feature matrix
    num_files = length(segment_features);
    num_features = length(all_feature_names) * 2; % mean and std
    X = zeros(num_files, num_features);
    Y = labels;
    feature_names = all_feature_names;
    
    % Calculate mean and std for each feature across segments
    for i = 1:num_files
        if isempty(segment_features{i})
            X(i,:) = NaN;
            continue;
        end
        
        % Convert struct array to matrix
        feature_matrix = [];
        for j = 1:length(all_feature_names)
            feature_name = all_feature_names{j};
            values = [segment_features{i}.(feature_name)];
            feature_matrix = [feature_matrix; values];
        end
        feature_matrix = feature_matrix';
        
        % Calculate statistics
        mean_vals = mean(feature_matrix, 1, 'omitnan');
        std_vals = std(feature_matrix, 0, 1, 'omitnan');
        X(i,:) = [mean_vals, std_vals];
    end
    
    % Remove invalid rows
    valid_rows = ~any(isnan(X), 2) & ~strcmp(Y, 'UNLABELED');
    X = X(valid_rows,:);
    Y = Y(valid_rows);
end

function [X, Y, feature_names] = consolidate_lung_features(segment_features, labels)
  % Get feature names from first non-empty entry
    feature_names = {};
    for i = 1:length(segment_features)
        if ~isempty(segment_features{i}) && ~isempty(fieldnames(segment_features{i}))
            feature_names = fieldnames(segment_features{i});
            break;
        end
    end
    
    % Initialize feature matrix
    num_files = length(segment_features);
    num_features = length(feature_names);
    X = zeros(num_files, num_features);
    Y = labels;
    
    % Populate feature matrix
    for i = 1:num_files
        if isempty(segment_features{i}) || isempty(fieldnames(segment_features{i}))
            X(i,:) = NaN;
            continue;
        end
        
        for j = 1:num_features
            feature_name = feature_names{j};
            val = segment_features{i}.(feature_name);
            if isscalar(val)
                X(i,j) = val;
            else
                X(i,j) = mean(val(:), 'omitnan');
            end
        end
    end
    
    % Remove invalid rows
    valid_rows = ~any(isnan(X), 2) & ~strcmp(Y, 'UNLABELED');
    X = X(valid_rows,:);
    Y = Y(valid_rows);
end
function features = extractHeartSegmentFeatures(segment_audio, Fs)
    % Extracts features relevant for heart sound segments
    
    % Initialize features structure with default NaN values
    features = struct(...
        'Energy_LowFreq', NaN, ...
        'Energy_MidFreq', NaN, ...
        'Energy_HighFreq', NaN, ...
        'SpectralFlatness', NaN, ...
        'SpectralEntropy', NaN, ...
        'RMS', NaN);
    
    if isempty(segment_audio) || all(segment_audio == 0)
        return;
    end

    % Ensure column vector
    segment_audio = segment_audio(:);

    % 1. Energy in Frequency Bands
    f_low_band = [20, 100];   % S1/S2 dominant frequencies
    f_mid_band = [100, 300];  % Murmur frequencies
    f_high_band = [300, 800]; % High frequency components
    
    % Design bandpass filters
    [b_low, a_low] = butter(4, f_low_band/(Fs/2), 'bandpass');
    [b_mid, a_mid] = butter(4, f_mid_band/(Fs/2), 'bandpass');
    [b_high, a_high] = butter(4, f_high_band/(Fs/2), 'bandpass');
    
    % Filter and calculate energy
    sig_low = filtfilt(b_low, a_low, segment_audio);
    sig_mid = filtfilt(b_mid, a_mid, segment_audio);
    sig_high = filtfilt(b_high, a_high, segment_audio);
    
    features.Energy_LowFreq = sum(sig_low.^2);
    features.Energy_MidFreq = sum(sig_mid.^2);
    features.Energy_HighFreq = sum(sig_high.^2);

    % 2. Spectral Features
    N = length(segment_audio);
    Y = fft(segment_audio);
    P2 = abs(Y/N);
    half_len = floor(N/2)+1;
    P1 = P2(1:half_len);
    P1(2:end-1) = 2*P1(2:end-1);
    
    % Spectral Flatness
    if all(P1 > 0)
        geometric_mean = exp(mean(log(P1)));
        arithmetic_mean = mean(P1);
        features.SpectralFlatness = geometric_mean / arithmetic_mean;
    end
    
    % Spectral Entropy
    Pxx_norm = P1 / sum(P1);
    Pxx_norm(Pxx_norm == 0) = eps; % Avoid log(0)
    features.SpectralEntropy = -sum(Pxx_norm .* log2(Pxx_norm));

    % 3. RMS Energy
    features.RMS = rms(segment_audio);
end
function features = extractLungSoundFeatures(audio_segment, Fs)
    % Initialize with default values
    features = struct(...
        'RMS', NaN, ...
        'ZCR', NaN, ...
        'SpectralCentroid', NaN, ...
        'SpectralSpread', NaN, ...
        'SpectralFlatness', NaN, ...
        'PeakFrequency', NaN, ...
        'Impulsiveness', NaN);
    
    if isempty(audio_segment) || all(audio_segment == 0)
        return;
    end

    % Ensure column vector
    audio_segment = audio_segment(:);
    
    % 1. Basic Time Domain Features
    features.RMS = rms(audio_segment);
    features.ZCR = sum(abs(diff(sign(audio_segment)))) / (2 * length(audio_segment));
    
    % 2. Frequency Domain Features
    N = length(audio_segment);
    Y = fft(audio_segment);
    P2 = abs(Y/N);
    half_len = floor(N/2)+1;
    P1 = P2(1:half_len);
    P1(2:end-1) = 2*P1(2:end-1);
    freq_axis = (0:half_len-1) * (Fs/N);
    
    % Spectral Centroid and Spread
    if sum(P1) > 0
        features.SpectralCentroid = sum(freq_axis(:) .* P1(:)) / sum(P1);
        features.SpectralSpread = sqrt(sum(((freq_axis(:) - features.SpectralCentroid).^2) .* P1(:)) / sum(P1));
    end
    
    % Spectral Flatness
    if all(P1 > 0)
        features.SpectralFlatness = exp(mean(log(P1))) / mean(P1);
    end
    
    % Peak Frequency
    [~, idx] = max(P1);
    features.PeakFrequency = freq_axis(idx);
    
    % Impulsiveness (for crackles)
    features.Impulsiveness = max(abs(audio_segment)) / mean(abs(audio_segment));
end
function export_features(X_heart, Y_heart, X_lung, Y_lung, export_dir)
    % Ensure export directory exists
    if ~exist(export_dir, 'dir')
        mkdir(export_dir);
    end
    
    % Export heart sound features
    heart_features_file = fullfile(export_dir, 'heart_features.csv');
    heart_labels_file = fullfile(export_dir, 'heart_labels.csv');
    
    % Write numeric features
    writematrix(X_heart, heart_features_file);
    
    % Write labels (ensure it's a column vector)
    if isrow(Y_heart)
        Y_heart = Y_heart';
    end
    writecell(Y_heart, heart_labels_file);
    
    % Export lung sound features
    lung_features_file = fullfile(export_dir, 'lung_features.csv');
    lung_labels_file = fullfile(export_dir, 'lung_labels.csv');
    
    % Write numeric features
    writematrix(X_lung, lung_features_file);
    
    % Write labels (ensure it's a column vector)
    if isrow(Y_lung)
        Y_lung = Y_lung';
    end
    writecell(Y_lung, lung_labels_file);
    
    fprintf('Features exported to:\n%s\n%s\n%s\n%s\n', ...
            heart_features_file, heart_labels_file, ...
            lung_features_file, lung_labels_file);
end

function create_visualizations(heart_data, lung_data, ...
                             mfccs_heart, mfccs_lung, ...
                             X_heart, Y_heart, ...
                             X_lung, Y_lung, ...
                             heart_feature_names, lung_feature_names, ...
                             Fs_heart, Fs_lung, ...
                             heart_filenames, export_dir)
    
    %% 1. Heart Sound Comparison
    fig1 = figure('Position', [100, 100, 1200, 800], 'Name', 'Heart Sound Comparison');
    
    % Find one normal and one abnormal sample
    normal_idx = find(strcmp(Y_heart, 'normal'), 1);
    abnormal_idx = find(strcmp(Y_heart, 'abnormal'), 1);
    
    % Time domain
    subplot(2,3,1);
    plot((0:length(heart_data{normal_idx})-1)/Fs_heart, heart_data{normal_idx});
    title('Normal Heart Sound (Time)');
    xlabel('Time (s)'); ylabel('Amplitude');
    
    subplot(2,3,4);
    plot((0:length(heart_data{abnormal_idx})-1)/Fs_heart, heart_data{abnormal_idx});
    title('Abnormal Heart Sound (Time)');
    xlabel('Time (s)'); ylabel('Amplitude');
    
    % Spectrogram
    subplot(2,3,2);
    spectrogram(heart_data{normal_idx}, 512, 256, 512, Fs_heart, 'yaxis');
    title('Normal Spectrogram');
    ylim([20 1000]);
    
    subplot(2,3,5);
    spectrogram(heart_data{abnormal_idx}, 512, 256, 512, Fs_heart, 'yaxis');
    title('Abnormal Spectrogram');
    ylim([20 1000]);
    
    % MFCCs
    subplot(2,3,3);
    imagesc(mfccs_heart{normal_idx}');
    title('Normal MFCCs');
    xlabel('Frame'); ylabel('Coefficient');
    colorbar;
    
    subplot(2,3,6);
    imagesc(mfccs_heart{abnormal_idx}');
    title('Abnormal MFCCs');
    xlabel('Frame'); ylabel('Coefficient');
    colorbar;
    
    %% 2. Feature Distributions
    fig2 = figure('Position', [100, 100, 1200, 600]);
    
    % Heart sound features
    subplot(1,2,1);
    features_to_plot = {'RMS_Mean', 'ZCR_Mean', 'SpecCentroid_Mean'};
    feature_data = [];
    groups = [];
    for i = 1:length(features_to_plot)
        feat_idx = find(strcmp(heart_feature_names, features_to_plot{i}));
        feature_data = [feature_data; X_heart(:,feat_idx)];
        groups = [groups; i*ones(size(X_heart,1),1)];
    end
    boxplot(feature_data, groups);
    set(gca, 'XTickLabel', features_to_plot);
    title('Heart Sound Feature Distribution');
    ylabel('Normalized Value');
    grid on;
    
    % Lung sound features
    subplot(1,2,2);
    features_to_plot = {'RMS', 'ZCR', 'SpectralCentroid'};
    feature_data = [];
    groups = [];
    for i = 1:length(features_to_plot)
        feat_idx = find(strcmp(lung_feature_names, features_to_plot{i}));
        feature_data = [feature_data; X_lung(:,feat_idx)];
        groups = [groups; i*ones(size(X_lung,1),1)];
    end
    boxplot(feature_data, groups);
    set(gca, 'XTickLabel', features_to_plot);
    title('Lung Sound Feature Distribution');
    ylabel('Normalized Value');
    grid on;
    
    %% 3. Heart Sound Segmentation Example
    fig3 = figure('Position', [100, 100, 1200, 400]);
    sample_idx = 1; % Change to view different samples
    sig = heart_data{sample_idx};
    [S1_locs, S2_locs, systole, diastole] = segmentHeartSounds(sig, Fs_heart);
    
    t = (0:length(sig)-1)/Fs_heart;
    plot(t, sig);
    hold on;
    plot(S1_locs/Fs_heart, sig(S1_locs), 'ro', 'MarkerSize', 10);
    plot(S2_locs/Fs_heart, sig(S2_locs), 'go', 'MarkerSize', 10);
    for i = 1:size(systole,1)
        rectangle('Position', [systole(i,1)/Fs_heart, min(sig), ...
            (systole(i,2)-systole(i,1))/Fs_heart, max(sig)-min(sig)], ...
            'FaceColor', [1 0 0 0.1], 'EdgeColor', 'none');
    end
    for i = 1:size(diastole,1)
        rectangle('Position', [diastole(i,1)/Fs_heart, min(sig), ...
            (diastole(i,2)-diastole(i,1))/Fs_heart, max(sig)-min(sig)], ...
            'FaceColor', [0 1 0 0.1], 'EdgeColor', 'none');
    end
    title(sprintf('Heart Sound Segmentation: %s', heart_filenames{sample_idx}));
    xlabel('Time (s)'); ylabel('Amplitude');
    legend('Signal', 'S1', 'S2', 'Systole', 'Diastole');
    grid on;
    
    %% 4. Class Distributions
    fig4 = figure('Position', [100, 100, 800, 400]);
    
    % Heart sounds
    subplot(1,2,1);
    heart_counts = countcats(categorical(Y_heart));
    pie(heart_counts);
    title('Heart Sound Class Distribution');
    legend(categories(categorical(Y_heart)), 'Location', 'eastoutside');
    
    % Lung sounds
    subplot(1,2,2);
    lung_counts = countcats(categorical(Y_lung));
    pie(lung_counts);
    title('Lung Sound Class Distribution');
    legend(categories(categorical(Y_lung)), 'Location', 'eastoutside');
    
    %% 5. Feature Correlations
    fig5 = figure('Position', [100, 100, 1000, 800]);
    
    % Heart features
    subplot(1,2,1);
    corr_matrix = corr(X_heart, 'Rows', 'complete');
    imagesc(corr_matrix);
    colorbar;
    title('Heart Sound Feature Correlation');
    xticks(1:length(heart_feature_names));
    yticks(1:length(heart_feature_names));
    xticklabels(heart_feature_names);
    yticklabels(heart_feature_names);
    xtickangle(45);
    set(gca, 'FontSize', 8);
    
    % Lung features
    subplot(1,2,2);
    corr_matrix = corr(X_lung, 'Rows', 'complete');
    imagesc(corr_matrix);
    colorbar;
    title('Lung Sound Feature Correlation');
    xticks(1:length(lung_feature_names));
    yticks(1:length(lung_feature_names));
    xticklabels(lung_feature_names);
    yticklabels(lung_feature_names);
    xtickangle(45);
    set(gca, 'FontSize', 8);
    
    %% Save all figures
    saveas(fig1, fullfile(export_dir, 'heart_comparison.png'));
    saveas(fig2, fullfile(export_dir, 'feature_distributions.png'));
    saveas(fig3, fullfile(export_dir, 'heart_segmentation.png'));
    saveas(fig4, fullfile(export_dir, 'class_distributions.png'));
    saveas(fig5, fullfile(export_dir, 'feature_correlations.png'));
end