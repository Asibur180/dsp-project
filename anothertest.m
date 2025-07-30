% Set the base data directory (adjust this to your actual path)
data_dir = 'D:\dsp project\Data';
export_dir = 'D:\DSP Project\Data';
% --- Loading ALL Heart Sound Files ---
heart_sounds_folder = fullfile(data_dir, 'HeartSounds\physionet');
heart_files = dir(fullfile(heart_sounds_folder, '*.wav')); % Get all .wav files in the folder

% Initialize cell arrays to store data and Fs for all files (optional, but good practice)
all_heart_data = cell(length(heart_files), 1);
all_Fs_heart = zeros(length(heart_files), 1);
all_heart_filenames = cell(length(heart_files), 1);

fprintf('--- Loading ALL Heart Sound Files ---\n');
for i = 1:length(heart_files)
    current_filename = heart_files(i).name;
    current_filepath = fullfile(heart_sounds_folder, current_filename);

    if exist(current_filepath, 'file') == 2
        [data, Fs] = audioread(current_filepath);
        all_heart_data{i} = data;
        all_Fs_heart(i) = Fs;
        all_heart_filenames{i} = current_filename;

        fprintf('  Loaded: %s (Length: %.2f s, Fs: %d Hz)\n', current_filename, length(data)/Fs, Fs);
    else
        warning('Heart sound file not found: %s\n', current_filepath);
    end
end
fprintf('Finished loading %d heart sound files.\n\n', length(heart_files));

% --- Loading ALL Lung Sound Files ---
lung_sounds_folder = fullfile(data_dir, 'LungSounds');
lung_files = dir(fullfile(lung_sounds_folder, '*.wav')); % Get all .wav files in the folder

% Initialize cell arrays to store data and Fs for all files
all_lung_data = cell(length(lung_files), 1);
all_Fs_lung = zeros(length(lung_files), 1);
all_lung_filenames = cell(length(lung_files), 1);

fprintf('--- Loading ALL Lung Sound Files ---\n');
for i = 1:length(lung_files)
    current_filename = lung_files(i).name;
    current_filepath = fullfile(lung_sounds_folder, current_filename);

    if exist(current_filepath, 'file') == 2
        [data, Fs] = audioread(current_filepath);
        all_lung_data{i} = data;
        all_Fs_lung(i) = Fs;
        all_lung_filenames{i} = current_filename;

        fprintf('  Loaded: %s (Length: %.2f s, Fs: %d Hz)\n', current_filename, length(data)/Fs, Fs);
    else
        warning('Lung sound file not found: %s\n', current_filepath);
    end
end
fprintf('Finished loading %d lung sound files.\n', length(lung_files));

% Now, all_heart_data and all_lung_data cell arrays contain the audio data for each file.
% You can access them like:
% first_heart_audio = all_heart_data{1};
% first_heart_Fs = all_Fs_heart(1);
% --- 1. Resampling All Loaded Data ---

% Choose a target sampling rate. Let's aim for a common practical Fs.
% Heart sounds usually don't exceed 1000-2000 Hz. Lung sounds can go a bit higher (2000-3000 Hz).
% 4000 Hz or 8000 Hz is a good balance for typical stethoscopic audio.
target_Fs = 4000; % Hz

fprintf('--- Resampling All Audio Files to %d Hz ---\n', target_Fs);

% Process Heart Sounds
processed_heart_data = cell(size(all_heart_data));
for i = 1:length(all_heart_data)
    current_data = all_heart_data{i};
    current_Fs = all_Fs_heart(i);
    
    if current_Fs ~= target_Fs
        processed_heart_data{i} = resample(current_data, target_Fs, current_Fs);
        fprintf('  Resampled Heart Sound "%s" from %d Hz to %d Hz\n', ...
            all_heart_filenames{i}, current_Fs, target_Fs);
    else
        processed_heart_data{i} = current_data;
        fprintf('  Heart Sound "%s" already at %d Hz\n', all_heart_filenames{i}, target_Fs);
    end
end

% Process Lung Sounds
processed_lung_data = cell(size(all_lung_data));
for i = 1:length(all_lung_data)
    current_data = all_lung_data{i};
    current_Fs = all_Fs_lung(i);
    
    if current_Fs ~= target_Fs
        processed_lung_data{i} = resample(current_data, target_Fs, current_Fs);
        fprintf('  Resampled Lung Sound "%s" from %d Hz to %d Hz\n', ...
            all_lung_filenames{i}, current_Fs, target_Fs);
    else
        processed_lung_data{i} = current_data;
        fprintf('  Lung Sound "%s" already at %d Hz\n', all_lung_filenames{i}, target_Fs);
    end
end
fprintf('Finished resampling.\n\n');

% Update the Fs variables to the target_Fs for future steps
Fs_heart_processed = target_Fs; 
Fs_lung_processed = target_Fs;
% --- 2. Filtering All Processed Data ---

% Design filters
% For heart sounds (20-1000 Hz approx) - 4th order Butterworth
f_low_heart = 20;
f_high_heart = 1000;
[b_h, a_h] = butter(4, [f_low_heart f_high_heart]/(Fs_heart_processed/2), 'bandpass');

% For lung sounds (100-2000 Hz approx) - 4th order Butterworth
f_low_lung = 100;
f_high_lung = 1999; % Consider slightly higher for crackles if needed (e.g., 2500-3000 Hz)
[b_l, a_l] = butter(4, [f_low_lung f_high_lung]/(Fs_lung_processed/2), 'bandpass');

% Optional: Notch filter for 50/60 Hz power line noise if present and problematic
% d = designfilt('bandstopiir','FilterOrder',2, ...
%                'HalfPowerFrequency1',49,'HalfPowerFrequency2',51, ...
%                'DesignMethod','butter','Samplerate',target_Fs);

filtered_heart_data = cell(size(processed_heart_data));
fprintf('--- Filtering All Audio Files ---\n');
for i = 1:length(processed_heart_data)
    % Apply filter. filtfilt applies zero-phase filtering (no phase distortion).
    filtered_heart_data{i} = filtfilt(b_h, a_h, processed_heart_data{i});
    % Optional: filtered_heart_data{i} = filtfilt(d, filtered_heart_data{i}); % Apply notch
    fprintf('  Filtered Heart Sound: %s\n', all_heart_filenames{i});
end

filtered_lung_data = cell(size(processed_lung_data));
for i = 1:length(processed_lung_data)
    % Apply filter
    filtered_lung_data{i} = filtfilt(b_l, a_l, processed_lung_data{i});
    % Optional: filtered_lung_data{i} = filtfilt(d, filtered_lung_data{i}); % Apply notch
    fprintf('  Filtered Lung Sound: %s\n', all_lung_filenames{i});
end
fprintf('Finished filtering.\n\n');
% --- 3. Normalization All Filtered Data ---

normalized_heart_data = cell(size(filtered_heart_data));
fprintf('--- Normalizing All Filtered Audio Files ---\n');
for i = 1:length(filtered_heart_data)
    current_data = filtered_heart_data{i};
    if ~isempty(current_data) && max(abs(current_data)) > 0
        normalized_heart_data{i} = current_data / max(abs(current_data));
    else
        normalized_heart_data{i} = current_data; % Handle empty or zero signals
    end
    fprintf('  Normalized Heart Sound: %s\n', all_heart_filenames{i});
end

normalized_lung_data = cell(size(filtered_lung_data));
for i = 1:length(filtered_lung_data)
    current_data = filtered_lung_data{i};
    if ~isempty(current_data) && max(abs(current_data)) > 0
        normalized_lung_data{i} = current_data / max(abs(current_data));
    else
        normalized_lung_data{i} = current_data; % Handle empty or zero signals
    end
    fprintf('  Normalized Lung Sound: %s\n', all_lung_filenames{i});
end
fprintf('Finished normalization.\n\n');
% Create a figure showing the complete processing pipeline
figure('Position', [100, 100, 1200, 600], 'Name', 'Signal Processing Pipeline');

subplot(2,4,1);
plot((0:length(all_heart_data{1})-1)/all_Fs_heart(1), all_heart_data{1});
title('Raw Heart Sound');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,4,2);
plot((0:length(processed_heart_data{1})-1)/Fs_heart_processed, processed_heart_data{1});
title('Resampled (4000 Hz)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,4,3);
plot((0:length(filtered_heart_data{1})-1)/Fs_heart_processed, filtered_heart_data{1});
title('Bandpass Filtered (20-1000 Hz)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,4,4);
plot((0:length(normalized_heart_data{1})-1)/Fs_heart_processed, normalized_heart_data{1});
title('Normalized');
xlabel('Time (s)'); ylabel('Amplitude');

% Repeat for lung sounds
subplot(2,4,5);
plot((0:length(all_lung_data{1})-1)/all_Fs_lung(1), all_lung_data{1});
title('Raw Lung Sound');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,4,6);
plot((0:length(processed_lung_data{1})-1)/Fs_lung_processed, processed_lung_data{1});
title('Resampled (4000 Hz)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,4,7);
plot((0:length(filtered_lung_data{1})-1)/Fs_lung_processed, filtered_lung_data{1});
title('Bandpass Filtered (100-2000 Hz)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,4,8);
plot((0:length(normalized_lung_data{1})-1)/Fs_lung_processed, normalized_lung_data{1});
title('Normalized');
xlabel('Time (s)'); ylabel('Amplitude');

saveas(gcf, fullfile(export_dir, 'processing_pipeline.png'));
% --- 4. Feature Extraction (MFCCs Example) ---

% MFCC parameters (can be tuned)
% num_mfccs = 13; % Number of MFCC coefficients - THIS VARIABLE IS NO LONGER USED DIRECTLY IN CONSTRUCTOR
window_length_mfcc = 0.025; % Window length in seconds (e.g., 25 ms)
overlap_length_mfcc = 0.01;  % Overlap length in seconds (e.g., 10 ms)

% Ensure these parameters are consistent with your target_Fs
frame_len_samples = round(window_length_mfcc * target_Fs);
hop_len_samples = round(overlap_length_mfcc * target_Fs);

% Initialize cell arrays to store MFCCs
mfccs_heart = cell(size(normalized_heart_data));
mfccs_lung = cell(size(normalized_lung_data));

fprintf('--- Extracting MFCCs from All Audio Files ---\n');

% Heart Sound MFCCs
for i = 1:length(normalized_heart_data)
    current_data = normalized_heart_data{i};
    if ~isempty(current_data)
        % Ensure data is a column vector
        if size(current_data, 2) > size(current_data, 1)
            current_data = current_data';
        end
        
        % Use audioFeatureExtractor for robust feature extraction (requires Audio Toolbox)
        % *** REMOVED 'FeatureExtractionMode' AND 'NumCoefficients' ***
        afe = audioFeatureExtractor( ...
            'SampleRate', target_Fs, ...
            'Window', hann(frame_len_samples, 'periodic'), ...
            'OverlapLength', hop_len_samples, ...
            'mfcc', true); % Just enable MFCCs, number of coeffs is default
        
        current_mfccs = extract(afe, current_data);
        
        % If you still need a specific number of coefficients (e.g., 13),
        % you would truncate or select them *after* extraction:
        num_mfccs_desired = 13; % Define your desired number here
        if size(current_mfccs, 2) >= num_mfccs_desired
            current_mfccs = current_mfccs(:, 1:num_mfccs_desired);
        else
            warning('Not enough MFCCs extracted for %s. Expected %d, got %d.', ...
                     all_heart_filenames{i}, num_mfccs_desired, size(current_mfccs, 2));
            % You might need to pad with zeros or handle this case based on your needs
        end


        mfccs_heart{i} = current_mfccs; % mfccs will be a matrix (frames x coefficients)
        fprintf('  Extracted MFCCs for Heart Sound: %s (Size: %s)\n', ...
            all_heart_filenames{i}, mat2str(size(current_mfccs)));
    else
        mfccs_heart{i} = [];
        fprintf('  Skipped MFCC extraction for empty Heart Sound: %s\n', all_heart_filenames{i});
    end
end

% Lung Sound MFCCs
for i = 1:length(normalized_lung_data)
    current_data = normalized_lung_data{i};
    if ~isempty(current_data)
        % Ensure data is a column vector
        if size(current_data, 2) > size(current_data, 1)
            current_data = current_data';
        end
        
        % *** REMOVED 'FeatureExtractionMode' AND 'NumCoefficients' ***
        afe = audioFeatureExtractor( ...
            'SampleRate', target_Fs, ...
            'Window', hann(frame_len_samples, 'periodic'), ...
            'OverlapLength', hop_len_samples, ...
            'mfcc', true); % Just enable MFCCs, number of coeffs is default
        
        current_mfccs = extract(afe, current_data);

        % If you still need a specific number of coefficients (e.g., 13),
        % you would truncate or select them *after* extraction:
        num_mfccs_desired = 13; % Define your desired number here
        if size(current_mfccs, 2) >= num_mfccs_desired
            current_mfccs = current_mfccs(:, 1:num_mfccs_desired);
        else
            warning('Not enough MFCCs extracted for %s. Expected %d, got %d.', ...
                     all_lung_filenames{i}, num_mfccs_desired, size(current_mfccs, 2));
            % You might need to pad with zeros or handle this case based on your needs
        end

        mfccs_lung{i} = current_mfccs;
        fprintf('  Extracted MFCCs for Lung Sound: %s (Size: %s)\n', ...
            all_lung_filenames{i}, mat2str(size(current_mfccs)));
    else
        mfccs_lung{i} = [];
        fprintf('  Skipped MFCC extraction for empty Lung Sound: %s\n', all_lung_filenames{i});
    end
end
fprintf('Finished MFCC extraction.\n\n');
% --- 5. Extracting Other Features (RMS, ZCR, Spectral Centroid) ---

% Function to calculate features for a single segment (you'd loop through frames later)
% For simplicity here, we'll calculate for the whole audio clip first.
feature_segment_length = 0.05; % seconds
feature_overlap_length = 0.025; % seconds

heart_features_summary = struct('filename', {}, 'RMS_Mean', {}, 'ZCR_Mean', {}, 'SpecCentroid_Mean', {});
lung_features_summary = struct('filename', {}, 'RMS_Mean', {}, 'ZCR_Mean', {}, 'SpecCentroid_Mean', {});

fprintf('--- Extracting Summary Features (RMS, ZCR, Spec. Centroid) ---\n');

% Heart Sound Features
for i = 1:length(normalized_heart_data)
    current_data = normalized_heart_data{i};
    if isempty(current_data) || all(current_data == 0)
        heart_features_summary(i).filename = all_heart_filenames{i};
        heart_features_summary(i).RMS_Mean = NaN;
        heart_features_summary(i).ZCR_Mean = NaN;
        heart_features_summary(i).SpecCentroid_Mean = NaN;
        fprintf('  Skipped feature extraction for empty Heart Sound: %s\n', all_heart_filenames{i});
        continue;
    end
    
    % Calculate RMS (Root Mean Square)
    rms_val = rms(current_data);

    % Calculate Zero-Crossing Rate (ZCR)
    zcr_val = sum(abs(diff(sign(current_data)))) / (2 * length(current_data));

    % Calculate Spectral Centroid (requires FFT/spectrum)
    % A simple estimate for the whole signal
    N = length(current_data);
    Y = fft(current_data);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    freq_axis = (0:N/2) * (target_Fs/N);
    
    % Avoid division by zero if sum of P1 is zero
    if sum(P1) > 0
        spectral_centroid = sum(freq_axis' .* P1) / sum(P1);
    else
        spectral_centroid = 0;
    end

    heart_features_summary(i).filename = all_heart_filenames{i};
    heart_features_summary(i).RMS_Mean = rms_val;
    heart_features_summary(i).ZCR_Mean = zcr_val;
    heart_features_summary(i).SpecCentroid_Mean = spectral_centroid;
    fprintf('  Heart Sound: %s, RMS: %.4f, ZCR: %.4f, SpecCentroid: %.2f Hz\n', ...
            all_heart_filenames{i}, rms_val, zcr_val, spectral_centroid);
end

% Lung Sound Features
for i = 1:length(normalized_lung_data)
    current_data = normalized_lung_data{i};
    if isempty(current_data) || all(current_data == 0)
        lung_features_summary(i).filename = all_lung_filenames{i};
        lung_features_summary(i).RMS_Mean = NaN;
        lung_features_summary(i).ZCR_Mean = NaN;
        lung_features_summary(i).SpecCentroid_Mean = NaN;
        fprintf('  Skipped feature extraction for empty Lung Sound: %s\n', all_lung_filenames{i});
        continue;
    end

    rms_val = rms(current_data);
    zcr_val = sum(abs(diff(sign(current_data)))) / (2 * length(current_data));

    N = length(current_data);
    Y = fft(current_data);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    freq_axis = (0:N/2) * (target_Fs/N);

    if sum(P1) > 0
        spectral_centroid = sum(freq_axis' .* P1) / sum(P1);
    else
        spectral_centroid = 0;
    end

    lung_features_summary(i).filename = all_lung_filenames{i};
    lung_features_summary(i).RMS_Mean = rms_val;
    lung_features_summary(i).ZCR_Mean = zcr_val;
    lung_features_summary(i).SpecCentroid_Mean = spectral_centroid;
    fprintf('  Lung Sound: %s, RMS: %.4f, ZCR: %.4f, SpecCentroid: %.2f Hz\n', ...
            all_lung_filenames{i}, rms_val, zcr_val, spectral_centroid);
end
fprintf('Finished summary feature extraction.\n\n');
% --- Example Visualization of a Processed File ---

% Pick one heart sound and one lung sound to visualize (e.g., the first one)
if ~isempty(normalized_heart_data)
    sample_heart_idx = 1; % or choose another index
    sample_heart_signal = normalized_heart_data{sample_heart_idx};
    sample_heart_filename = all_heart_filenames{sample_heart_idx};

    figure('Name', 'Processed Heart Sound Example');
    subplot(2,1,1);
    plot((0:length(sample_heart_signal)-1)/Fs_heart_processed, sample_heart_signal);
    title(sprintf('Processed Heart Sound Waveform: %s', sample_heart_filename));
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;

    subplot(2,1,2);
    spectrogram(sample_heart_signal, round(0.03*Fs_heart_processed), round(0.015*Fs_heart_processed), [], Fs_heart_processed, 'yaxis');
    title(sprintf('Processed Heart Sound Spectrogram: %s', sample_heart_filename));
    xlabel('Time (s)'); ylabel('Frequency (Hz)'); colorbar; colormap(jet);
    drawnow; % Ensure plot updates
end

if ~isempty(normalized_lung_data)
    sample_lung_idx = 1; % or choose another index
    sample_lung_signal = normalized_lung_data{sample_lung_idx};
    sample_lung_filename = all_lung_filenames{sample_lung_idx};

    figure('Name', 'Processed Lung Sound Example');
    subplot(2,1,1);
    plot((0:length(sample_lung_signal)-1)/Fs_lung_processed, sample_lung_signal);
    title(sprintf('Processed Lung Sound Waveform: %s', sample_lung_filename));
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;

    subplot(2,1,2);
    spectrogram(sample_lung_signal, round(0.03*Fs_lung_processed), round(0.015*Fs_lung_processed), [], Fs_lung_processed, 'yaxis');
    title(sprintf('Processed Lung Sound Spectrogram: %s', sample_lung_filename));
    xlabel('Time (s)'); ylabel('Frequency (Hz)'); colorbar; colormap(jet);
    drawnow;
end
% --- Function for Heart Sound Segment Features ---
function [features] = extractHeartSegmentFeatures(segment_audio, Fs)
    % Extracts features relevant for murmur detection from a given audio segment.
    
    if isempty(segment_audio) || all(segment_audio == 0)
        features.Energy_LowFreq = NaN;
        features.Energy_MidFreq = NaN;
        features.Energy_HighFreq = NaN;
        features.SpectralFlatness = NaN;
        features.SpectralEntropy = NaN;
        features.RMS = NaN;
        return;
    end

    % Ensure column vector
    if size(segment_audio, 2) > size(segment_audio, 1)
        segment_audio = segment_audio';
    end

    % 1. Energy in Frequency Bands
    % Define bands (these are typical, can be tuned)
    f_low_band = [20, 100];   % S1/S2 dominant
    f_mid_band = [100, 300];  % Murmurs often here
    f_high_band = [300, 800]; % Higher frequency murmurs, clicks

    % Design simple bandpass filters for energy calculation
    [b_low, a_low] = butter(4, f_low_band / (Fs/2), 'bandpass');
    [b_mid, a_mid] = butter(4, f_mid_band / (Fs/2), 'bandpass');
    [b_high, a_high] = butter(4, f_high_band / (Fs/2), 'bandpass');

    sig_low = filtfilt(b_low, a_low, segment_audio);
    sig_mid = filtfilt(b_mid, a_mid, segment_audio);
    sig_high = filtfilt(b_high, a_high, segment_audio);

    features.Energy_LowFreq = sum(sig_low.^2);
    features.Energy_MidFreq = sum(sig_mid.^2);
    features.Energy_HighFreq = sum(sig_high.^2);

    % 2. Spectral Flatness and Entropy
    % Use pwelch to get Power Spectral Density (PSD)
    [Pxx, Freqs] = pwelch(segment_audio, [], [], [], Fs); % [] means default window, overlap, NFFT

    if all(Pxx == 0) || sum(log(Pxx)) == -inf % Handle zero/negative Pxx from noise or very short segments
        features.SpectralFlatness = 0; % Or NaN, depending on how you want to handle it
        features.SpectralEntropy = 0;
    else
        % Spectral Flatness Measure (SFM)
        geometric_mean_psd = exp(mean(log(Pxx)));
        arithmetic_mean_psd = mean(Pxx);
        features.SpectralFlatness = geometric_mean_psd / arithmetic_mean_psd; % Values 0-1 (1 is flat/noise-like)
        
        % Spectral Entropy
        % Normalize PSD to sum to 1 (treat as probability distribution)
        Pxx_norm = Pxx / sum(Pxx);
        Pxx_norm(Pxx_norm == 0) = eps; % Avoid log(0)
        features.SpectralEntropy = -sum(Pxx_norm .* log2(Pxx_norm));
    end

    % 3. RMS Energy of the segment
    features.RMS = rms(segment_audio);
end
% --- Function for Lung Sound Features ---
function [features] = extractLungSoundFeatures(audio_segment, Fs)
    % Extracts features relevant for crackle and wheeze detection.
    
    if isempty(audio_segment) || all(audio_segment == 0)
        features.RMS = NaN;
        features.ZCR = NaN;
        features.SpectralCentroid = NaN;
        features.SpectralSpread = NaN;
        features.SpectralFlatness = NaN; % for overall segment
        features.PeakFrequency = NaN;    % for wheezes
        features.Impulsiveness = NaN;    % for crackles
        return;
    end

    % Ensure column vector
    if size(audio_segment, 2) > size(audio_segment, 1)
        audio_segment = audio_segment';
    end

    % 1. General Time/Frequency Domain Features (useful for both)
    features.RMS = rms(audio_segment);
    features.ZCR = sum(abs(diff(sign(audio_segment)))) / (2 * length(audio_segment));

    % Spectral Centroid, Spread, Flatness (using a short-time FFT for better resolution)
    frame_len_samples = round(0.03 * Fs); % e.g., 30 ms frames
    hop_len_samples = round(0.015 * Fs); % 15 ms hop

    afe = audioFeatureExtractor( ...
        'SampleRate', Fs, ...
        'Window', hann(frame_len_samples, 'periodic'), ...
        'OverlapLength', hop_len_samples, ...
        'spectralCentroid', true, ...
        'spectralSpread', true, ...
        'spectralFlatness', true);
    
    features_per_frame = extract(afe, audio_segment);
    
    % Take mean/std over frames for overall segment features
    if ~isempty(features_per_frame)
        features.SpectralCentroid = mean(features_per_frame(:,1));
        features.SpectralSpread = mean(features_per_frame(:,2));
        features.SpectralFlatness = mean(features_per_frame(:,3));
    else % Handle very short segments that might not produce frames
        features.SpectralCentroid = NaN;
        features.SpectralSpread = NaN;
        features.SpectralFlatness = NaN;
    end

    % 2. Impulsiveness (for Crackles)
    % Ratio of peak absolute amplitude to mean absolute amplitude
    features.Impulsiveness = max(abs(audio_segment)) / mean(abs(audio_segment));

    % 3. Peak Frequency (for Wheezes) - More robust detection
    % This is a simple peak detection. For robust wheeze detection,
    % you'd look for sustained peaks across multiple frames of the spectrogram.
    [Pxx, Freqs] = pwelch(audio_segment, [], [], [], Fs);
    [max_power, idx_max_power] = max(Pxx);
    if max_power > 0 % Avoid division by zero/log(0) issues for very quiet segments
        features.PeakFrequency = Freqs(idx_max_power);
    else
        features.PeakFrequency = NaN;
    end
end
% --- Heart Sound Segmentation Function ---
function [S1_locs, S2_locs, systole_intervals, diastole_intervals] = segmentHeartSounds(heart_audio, Fs)
    % Segments heart sounds into S1, S2, systole, and diastole.
    % This is a simplified, envelope-based method. More robust algorithms exist.

    if isempty(heart_audio) || all(heart_audio == 0)
        S1_locs = []; S2_locs = []; systole_intervals = []; diastole_intervals = [];
        return;
    end

    % 1. Preprocessing for envelope extraction
    % Rectification and Low-Pass Filtering
    abs_audio = abs(heart_audio);
    
    % Design a low-pass filter to smooth the envelope
    cutoff_freq_envelope = 10; % Hz (e.g., 10-20 Hz)
    [b_lp, a_lp] = butter(4, cutoff_freq_envelope / (Fs/2), 'low');
    envelope = filtfilt(b_lp, a_lp, abs_audio);

    % Normalize envelope for consistent peak detection
    envelope = envelope / max(envelope);

    % 2. Peak Detection
    % Find peaks in the envelope. Tune MinPeakProminence and MinPeakDistance carefully.
    % MinPeakDistance: approximate minimum duration of a cardiac cycle or half-cycle
    min_peak_dist_samples = round(0.2 * Fs); % e.g., 0.2 seconds between major peaks

    [pks, locs] = findpeaks(envelope, 'MinPeakProminence', 0.2, 'MinPeakDistance', min_peak_dist_samples);
    
    if isempty(locs)
        S1_locs = []; S2_locs = []; systole_intervals = []; diastole_intervals = [];
        return;
    end

    % 3. Classify Peaks as S1 or S2 (Simplified Heuristic)
    % This is the tricky part. A common simple heuristic:
    % - S1 is typically louder than S2 in some leads.
    % - The S1-S2 interval (systole) is shorter than S2-S1 (diastole).
    % We'll use the latter: alternating short-long intervals.

    S1_locs = [];
    S2_locs = [];
    
    % Start with the first peak as S1 (arbitrary, could be wrong)
    % Then alternate based on amplitude or interval logic
    
    % A more robust approach might look at peak amplitudes and typical intervals
    % For a basic approach, let's assume alternating S1 and S2 based on interval length
    % Identify the two most prominent peak types (S1 and S2) by clustering peak amplitudes
    % For simplicity, let's just use a simple alternating pattern after a few initial peaks
    
    % Initialize by assuming the first two major peaks are S1 and S2,
    % then use the R-R interval from ECG or estimate cardiac cycle duration
    
    % Better heuristic: S1 is usually the loudest, or S1-S2 is shorter than S2-S1
    
    % Let's use a simplified approach where we identify two dominant peak "types"
    % based on amplitude, and then try to order them.
    
    % A common approach without ECG reference:
    % Identify all prominent peaks. Iterate and identify short vs. long intervals.
    % The shorter interval is systole (S1-S2), the longer is diastole (S2-S1).
    
    % Simplified S1/S2 Assignment (requires at least 2 peaks)
    if length(locs) < 2
        S1_locs = []; S2_locs = []; systole_intervals = []; diastole_intervals = [];
        return;
    end

    % Store candidate S1/S2 pairs and their interval lengths
    candidate_intervals = diff(locs);
    
    % Find an approximate median interval to classify short/long
    median_interval = median(candidate_intervals);

    for k = 1:length(locs) - 1
        current_interval = candidate_intervals(k);
        if current_interval < median_interval * 0.8 % Adjust threshold for short interval (systole)
            % This interval is likely systole (S1-S2)
            S1_locs = [S1_locs, locs(k)];
            S2_locs = [S2_locs, locs(k+1)];
        end
    end
    
    % If we have S1 and S2 locations, derive intervals
    systole_intervals = [];
    diastole_intervals = [];

    if ~isempty(S1_locs) && ~isempty(S2_locs)
        % Ensure S1 and S2 are paired correctly
        for k = 1:min(length(S1_locs), length(S2_locs))
            if S2_locs(k) > S1_locs(k) % S1 must come before S2
                systole_intervals = [systole_intervals; S1_locs(k), S2_locs(k)];
                
                % Find next S1 after current S2 to define diastole
                if k < length(S1_locs) % Make sure there's a next S1
                    next_S1_loc = S1_locs(k+1);
                    if next_S1_loc > S2_locs(k)
                        diastole_intervals = [diastole_intervals; S2_locs(k), next_S1_loc];
                    end
                end
            end
        end
    end
    
 % --- Inside segmentHeartSounds function, around line 623 ---

    % Filter out empty/invalid segments
    % ADD THIS CHECK:
    if ~isempty(systole_intervals)
        systole_intervals = systole_intervals( (systole_intervals(:,2) - systole_intervals(:,1)) > 0, :);
    end
    
    % ADD THIS CHECK:
    if ~isempty(diastole_intervals)
        diastole_intervals = diastole_intervals( (diastole_intervals(:,2) - diastole_intervals(:,1)) > 0, :);
    end

    % Ensure S1_locs and S2_locs are unique and ordered.
    S1_locs = unique(S1_locs);
    S2_locs = unique(S2_locs);
    S1_locs = sort(S1_locs);
    S2_locs = sort(S2_locs);
end % End of function
% --- Extracting Features from Heart Sounds Per Segment ---
all_heart_segment_features = cell(size(normalized_heart_data));
heart_labels_for_ml = {}; % To store 'normal_heart', 'murmur' etc.

fprintf('--- Segmenting and Extracting Features from Heart Sounds ---\n');

for i = 1:length(normalized_heart_data)
    current_heart_audio = normalized_heart_data{i};
    current_filename = all_heart_filenames{i};

    if isempty(current_heart_audio) || all(current_heart_audio == 0)
        all_heart_segment_features{i} = struct(); % Store empty struct
        fprintf('  Skipped empty Heart Sound: %s\n', current_filename);
        continue;
    end

    [S1_locs, S2_locs, systole_intervals, diastole_intervals] = segmentHeartSounds(current_heart_audio, Fs_heart_processed);
    
    fprintf('  Processing Heart Sound: %s\n', current_filename);
    fprintf('    Detected S1s: %d, S2s: %d\n', length(S1_locs), length(S2_locs));
    fprintf('    Systolic intervals: %d, Diastolic intervals: %d\n', size(systole_intervals,1), size(diastole_intervals,1));

    file_segment_features = []; % To store features for this file's segments

    % Extract features from Systolic Segments
    for j = 1:size(systole_intervals, 1)
        start_idx = systole_intervals(j, 1);
        end_idx = systole_intervals(j, 2);
        segment_audio = current_heart_audio(start_idx:end_idx);
        
        features = extractHeartSegmentFeatures(segment_audio, Fs_heart_processed);
        features.SegmentType = 'systole'; % Add segment type
        file_segment_features = [file_segment_features; features];
    end

    % Extract features from Diastolic Segments
    for j = 1:size(diastole_intervals, 1)
        start_idx = diastole_intervals(j, 1);
        end_idx = diastole_intervals(j, 2);
        segment_audio = current_heart_audio(start_idx:end_idx);
        
        features = extractHeartSegmentFeatures(segment_audio, Fs_heart_processed);
        features.SegmentType = 'diastole'; % Add segment type
        file_segment_features = [file_segment_features; features];
    end
    
    all_heart_segment_features{i} = file_segment_features;
    % For initial ML, you might average features across all segments for a single file feature vector
    % Or, you can classify segments individually and then aggregate.
    % Let's plan to average for a single per-file feature vector for simplicity.
end
fprintf('Finished heart sound segmentation and feature extraction.\n\n');


% --- Extracting Features from Lung Sounds (Full segments or respiratory cycles) ---
all_lung_segment_features = cell(size(normalized_lung_data));
lung_labels_for_ml = {}; % To store 'normal_lung', 'crackles', 'wheezes', 'both' etc.

fprintf('--- Extracting Features from Lung Sounds ---\n');
for i = 1:length(normalized_lung_data)
    current_lung_audio = normalized_lung_data{i};
    current_filename = all_lung_filenames{i};

    if isempty(current_lung_audio) || all(current_lung_audio == 0)
        all_lung_segment_features{i} = struct();
        fprintf('  Skipped empty Lung Sound: %s\n', current_filename);
        continue;
    end

    % For lung sounds, you can extract features over fixed-size windows (frames)
    % or, ideally, segment into respiratory cycles (inhalation/exhalation).
    % For simplicity in this step, let's extract features over the *entire* lung sound file
    % as a single "segment". For more robust detection, you'd frame it or segment cycles.
    
    features = extractLungSoundFeatures(current_lung_audio, Fs_lung_processed);
    all_lung_segment_features{i} = features; % Storing a struct directly for each file
    
    fprintf('  Extracted Lung Sound Features for: %s\n', current_filename);
end
fprintf('Finished lung sound feature extraction.\n\n');

%% --- Load and Assign Labels ---
heart_annotations_path = fullfile(data_dir, 'Online Appendix_training set.csv');
if exist(heart_annotations_path, 'file')
    heart_annotations = readtable(heart_annotations_path);
    fprintf('Loaded heart annotations from: %s\n', heart_annotations_path);
    
    % Initialize Y_heart_combined
    Y_heart_combined = cell(length(all_heart_filenames), 1);
    
    for i = 1:length(all_heart_filenames)
        [~, record_name, ~] = fileparts(all_heart_filenames{i});
        idx_anno = find(strcmp(heart_annotations.ChallengeRecordName, record_name), 1);
        
        if ~isempty(idx_anno)
            raw_label = heart_annotations.Class__1_normal1_abnormal_(idx_anno);
            if raw_label == -1
                Y_heart_combined{i} = 'normal';
            else
                Y_heart_combined{i} = 'abnormal';
            end
        else
            Y_heart_combined{i} = 'unmatched';
        end
    end
else
    warning('Heart sound annotations not found at: %s', heart_annotations_path);
    Y_heart_combined = repmat({'normal'}, length(normalized_heart_data), 1);
end

% Compare normal vs abnormal heart sounds
figure('Position', [100, 100, 1200, 800], 'Name', 'Heart Sound Comparison');

% Find one normal and one abnormal sample
normal_idx = find(strcmp(Y_heart_combined, 'normal'), 1);
abnormal_idx = find(strcmp(Y_heart_combined, 'abnormal'), 1);

% Time domain
subplot(2,3,1);
plot((0:length(normalized_heart_data{normal_idx})-1)/Fs_heart_processed, normalized_heart_data{normal_idx});
title('Normal Heart Sound (Time)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,3,4);
plot((0:length(normalized_heart_data{abnormal_idx})-1)/Fs_heart_processed, normalized_heart_data{abnormal_idx});
title('Abnormal Heart Sound (Time)');
xlabel('Time (s)'); ylabel('Amplitude');

% Spectrogram
subplot(2,3,2);
spectrogram(normalized_heart_data{normal_idx}, 512, 256, 512, Fs_heart_processed, 'yaxis');
title('Normal Spectrogram');
ylim([20 1000]);

subplot(2,3,5);
spectrogram(normalized_heart_data{abnormal_idx}, 512, 256, 512, Fs_heart_processed, 'yaxis');
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

saveas(gcf, fullfile(export_dir, 'heart_comparison.png'));


set(gca, 'XTickLabel', features_to_plot);
title('Heart Sound Feature Distribution');
ylabel('Normalized Value');
grid on;


% Visualize heart sound segmentation
figure('Position', [100, 100, 1200, 400]);
sample_idx = 1; % Change to view different samples
sig = normalized_heart_data{sample_idx};
[S1_locs, S2_locs, systole, diastole] = segmentHeartSounds(sig, Fs_heart_processed);

t = (0:length(sig)-1)/Fs_heart_processed;
plot(t, sig);
hold on;
plot(S1_locs/Fs_heart_processed, sig(S1_locs), 'ro', 'MarkerSize', 10);
plot(S2_locs/Fs_heart_processed, sig(S2_locs), 'go', 'MarkerSize', 10);
for i = 1:size(systole,1)
    rectangle('Position', [systole(i,1)/Fs_heart_processed, min(sig), ...
        (systole(i,2)-systole(i,1))/Fs_heart_processed, max(sig)-min(sig)], ...
        'FaceColor', [1 0 0 0.1], 'EdgeColor', 'none');
end
for i = 1:size(diastole,1)
    rectangle('Position', [diastole(i,1)/Fs_heart_processed, min(sig), ...
        (diastole(i,2)-diastole(i,1))/Fs_heart_processed, max(sig)-min(sig)], ...
        'FaceColor', [0 1 0 0.1], 'EdgeColor', 'none');
end
title(sprintf('Heart Sound Segmentation: %s', all_heart_filenames{sample_idx}));
xlabel('Time (s)'); ylabel('Amplitude');
legend('Signal', 'S1', 'S2', 'Systole', 'Diastole');
grid on;

saveas(gcf, fullfile(export_dir, 'heart_segmentation.png'));



% --- 3.1 Data Preparation for Machine Learning (Lung Sounds Example) ---
% This section creates a dummy annotations table if one doesn't already exist.
% REPLACE THIS WITH YOUR ACTUAL ANNOTATION LOADING LOGIC FOR REAL DATA!

% IMPORTANT: Ensure 'data_dir' is defined and points to your project's main data folder.
% For example: data_dir = 'C:\Users\YourUsername\Documents\VirtualStethoscopeProject\Data\';
% data_dir = '/Users/YourUsername/Documents/VirtualStethoscopeProject/Data/';
% If data_dir is not defined, define it here:
% if ~exist('data_dir', 'var') || isempty(data_dir)
%     data_dir = 'YOUR_PATH_TO_YOUR_DATA_FOLDER_HERE'; % <--- SET YOUR ACTUAL PATH HERE
%     if ~isfolder(data_dir)
%         error('Data directory not found. Please set `data_dir` to a valid path.');
%     end
% end



% --- Consolidating Heart Sound Features per File ---
fprintf('--- Consolidating Heart Sound Features per File ---\n');

% Load Real Heart Sound Annotations once before the loop
fprintf('\n--- Loading Real Heart Sound Annotations ---\n');

% Path to the downloaded training set CSV. Assume it's in data_dir for now.
% ADJUST THIS PATH IF YOUR 'Online Appendix_training set.csv' IS IN A SUBFOLDER!
real_heart_annotations_filepath = fullfile(data_dir, 'Online Appendix_training set.csv'); 

heart_annotations_raw = []; % Initialize as empty
if exist(real_heart_annotations_filepath, 'file')
    heart_annotations_raw = readtable(real_heart_annotations_filepath);
    % Add this line immediately after heart_annotations_raw = readtable(real_heart_annotations_filepath);
fprintf('Actual column names in heart_annotations_raw table:\n');
disp(heart_annotations_raw.Properties.VariableNames{5});
    fprintf('Successfully loaded heart annotations from: %s\n', real_heart_annotations_filepath);
    % Optional: Display actual column names to confirm (already did this step)
    % fprintf('Actual column names in heart_annotations_raw table:\n');
    % disp(heart_annotations_raw.Properties.VariableNames);
else
    warning('Real heart sound annotations file not found at %s. Heart sound labels will be "unmatched".', real_heart_annotations_filepath);
end


% Determine all unique feature names from the structures (e.g., Energy_LowFreq, SpectralFlatness)
all_heart_feature_names = {};
for i = 1:length(all_heart_segment_features)
    if ~isempty(all_heart_segment_features{i})
        all_heart_feature_names = union(all_heart_feature_names, fieldnames(all_heart_segment_features{i}));
    end
end
% Remove 'SegmentType' as it's not a numerical feature for averaging
all_heart_feature_names = setdiff(all_heart_feature_names, {'SegmentType'});

num_heart_files = length(all_heart_filenames);
num_avg_features = length(all_heart_feature_names) * 2; % For mean and std dev of each feature

X_heart_combined = zeros(num_heart_files, num_avg_features); % For mean and std of features
Y_heart_combined = cell(num_heart_files, 1);
heart_file_lookup = cell(num_heart_files, 1);

for i = 1:num_heart_files
    current_filename_full = all_heart_filenames{i}; % Full filename like 'a0001.wav'
    current_file_segments = all_heart_segment_features{i}; % This is an array of structs
    heart_file_lookup{i} = current_filename_full; % Store full filename for lookup

    if isempty(current_file_segments) || (isstruct(current_file_segments) && all(cellfun(@(x) all(isnan(struct2array(x))), {current_file_segments}))) % Check if segments are empty or all NaNs
        fprintf('  No valid segments or features for Heart Sound: %s. Filling with NaNs.\n', current_filename_full);
        X_heart_combined(i, :) = NaN;
        Y_heart_combined{i} = 'UNLABELED'; % Or a specific 'no_detection' label
        continue;
    end

    % Extract numerical values for each feature from all segments of the current file
    temp_features_matrix = [];
    for j = 1:length(all_heart_feature_names)
        feature_name = all_heart_feature_names{j};
        % Get all values for this feature across all segments in the current file
        feature_values_for_file = [current_file_segments.(feature_name)];
        temp_features_matrix = [temp_features_matrix; feature_values_for_file];
    end
    
    % Transpose to have segments as rows, features as columns
    temp_features_matrix = temp_features_matrix';
    
    % Calculate mean and std dev for each feature across segments for this file
    mean_features = mean(temp_features_matrix, 1, 'omitnan'); % Ignore NaNs from segments with no data
    std_features = std(temp_features_matrix, 0, 1, 'omitnan'); % Ignore NaNs

    % Concatenate mean and std dev to form the final feature vector for this file
    X_heart_combined(i, :) = [mean_features, std_features];
    % Boxplots of key features for different classes
figure('Position', [100, 100, 1200, 600]);

% Heart sound features
subplot(1,2,1);
features_to_plot = {'RMS_Mean', 'ZCR_Mean', 'SpecCentroid_Mean'};
feature_data = [];
groups = [];
for i = 1:length(features_to_plot)
    feat_idx = find(strcmp(all_heart_feature_names, features_to_plot{i}));
    feature_data = [feature_data; X_heart_combined(:,feat_idx)];
    groups = [groups; i*ones(size(X_heart_combined,1),1)];
end
boxplot(feature_data, groups);
    
    % --- NEW LABEL ASSIGNMENT LOGIC FOR HEART SOUNDS (with corrected column names) ---
    % Extract the 'Challenge record name' part (e.g., 'a0001' from 'a0001.wav')
    [~, record_name, ~] = fileparts(current_filename_full); 

    if ~isempty(heart_annotations_raw)
        % Find the matching row in the loaded annotations table using the CORRECTED COLUMN NAME
        idx_anno = find(strcmp(heart_annotations_raw.ChallengeRecordName, record_name), 1);

        if ~isempty(idx_anno)
            % Extract the binary class label (-1 or 1) using the CORRECTED COLUMN NAME
           raw_label = heart_annotations_raw.Class__1_normal1_abnormal_(idx_anno);
            
            % Convert -1 to 'normal' and 1 to 'abnormal'
            if raw_label == -1
                Y_heart_combined{i} = 'normal';
            elseif raw_label == 1
                Y_heart_combined{i} = 'abnormal';
            else
                % Handle unexpected values if any
                Y_heart_combined{i} = 'unknown_label';
                warning('Unexpected heart label value: %d for %s\n', raw_label, current_filename_full);
            end
        else
            Y_heart_combined{i} = 'unmatched';
            % fprintf('  Warning: No annotation found for heart file: %s\n', current_filename_full); % Uncomment for more verbosity
        end
    else
        Y_heart_combined{i} = 'unmatched_no_table'; % Label if annotation table itself wasn't loaded
    end
    % --- END NEW LABEL ASSIGNMENT LOGIC ---
    
    fprintf('  Processed Heart Sound: %s, Features size: %s, Label: %s\n', current_filename_full, mat2str(size(X_heart_combined(i,:))), Y_heart_combined{i});
end

% Remove any rows that ended up with NaNs or 'UNLABELED'
valid_heart_rows = ~any(isnan(X_heart_combined), 2) & ...
                   ~strcmp(Y_heart_combined, 'UNLABELED') & ...
                   ~strcmp(Y_heart_combined, 'unmatched') & ...
                   ~strcmp(Y_heart_combined, 'unknown_label') & ...
                   ~strcmp(Y_heart_combined, 'unmatched_no_table');

X_heart_combined = X_heart_combined(valid_heart_rows, :);
Y_heart_combined = Y_heart_combined(valid_heart_rows);
heart_file_lookup = heart_file_lookup(valid_heart_rows);

fprintf('Consolidated Heart Feature matrix size: %s, Label vector size: %s\n', mat2str(size(X_heart_combined)), mat2str(size(Y_heart_combined)));

% ... (The rest of your code, including Lung Sound preparation and export, should follow) .

% ... (The rest of your code, including Lung Sound preparation and export, should follow) ...
% --- Preparing Lung Sound Features per File ---
% --- Load the master lung annotations CSV ---
% This CSV file should have been generated either by extending your dummy data
% or by running a separate script like 'generate_lung_master_csv.m'.
lung_master_annotations_path = fullfile(data_dir, 'lung_annotations.csv'); 
% NOTE: If you named your master CSV 'lung_annotations_master.csv',
% change the line above to:
% lung_master_annotations_path = fullfile(data_dir, 'lung_annotations_master.csv');

if exist(lung_master_annotations_path, 'file')
    lung_master_annotations_table = readtable(lung_master_annotations_path);
    fprintf('Loaded master lung annotations from: %s\n', lung_master_annotations_path);
else
    error('Master lung annotations CSV not found at %s. Please ensure it exists and the path is correct.', lung_master_annotations_path);
end
fprintf('\n--- Preparing Lung Sound Features per File ---\n');

num_lung_files = length(all_lung_filenames);
lung_feature_names = fieldnames(all_lung_segment_features{1}); % Get feature names from the first struct

X_lung_combined = zeros(num_lung_files, length(lung_feature_names));
Y_lung_combined = cell(num_lung_files, 1); % From previous annotation loading
lung_file_lookup = cell(num_lung_files, 1);

% Ensure annotations_table exists (as per previous section's dummy table setup)
% if not, run the dummy table creation code again from the previous response.
% --- Preparing Lung Sound Features per File ---
fprintf('\n--- Preparing Lung Sound Features per File ---\n');

% --- Load the master lung annotations CSV ---
% This CSV file should have been generated either by extending your dummy data
% or by running a separate script like 'generate_lung_master_csv.m'.
% If you named your master CSV 'lung_annotations_master.csv',
% change the path below accordingly.
lung_annotations_master_path = fullfile(data_dir, 'lung_annotations.csv'); 

lung_annotations_master_table = []; % Initialize as empty
if exist(lung_master_annotations_path, 'file')
    lung_master_annotations_table = readtable(lung_master_annotations_path);
    fprintf('Loaded master lung annotations from: %s\n', lung_master_annotations_path);
else
    error('Master lung annotations CSV not found at %s. Please ensure it exists and the path is correct.', lung_master_annotations_path);
end


num_lung_files = length(all_lung_filenames);
% Ensure all_lung_segment_features is not empty before trying to get fieldnames
if isempty(all_lung_segment_features) || isempty(all_lung_segment_features{1}) || isempty(fieldnames(all_lung_segment_features{1}))
    warning('No lung segment features found. Skipping lung feature consolidation.');
    X_lung_combined = [];
    Y_lung_combined = {};
    lung_file_lookup = {};
else
    lung_feature_names = fieldnames(all_lung_segment_features{1}); % Get feature names from the first struct
    X_lung_combined = zeros(num_lung_files, length(lung_feature_names));
    Y_lung_combined = cell(num_lung_files, 1);
    lung_file_lookup = cell(num_lung_files, 1);

    initial_lung_samples = 0; % Initialize for counting

    for i = 1:num_lung_files
        current_filename_full = all_lung_filenames{i}; % e.g., '101_1b1_Al_sc_Meditron.wav'
        current_features_struct = all_lung_segment_features{i};
        lung_file_lookup{i} = current_filename_full;
        initial_lung_samples = initial_lung_samples + 1; % Increment count for each file processed

        if isempty(fieldnames(current_features_struct))
            fprintf('  No features for Lung Sound: %s. Filling with NaNs.\n', current_filename_full);
            X_lung_combined(i, :) = NaN;
            Y_lung_combined{i} = 'UNLABELED_NO_FEATURES';
            continue; % Skip to next file if no features
        end

        % Populate feature vector X for the current file
        for j = 1:length(lung_feature_names)
            feature_val = current_features_struct.(lung_feature_names{j});
            if isscalar(feature_val)
                X_lung_combined(i, j) = feature_val;
            else 
                % This case might happen if extractLungSoundFeatures sometimes returns non-scalars
                % (e.g., if you added MFCCs directly without averaging them within the function).
                % Taking the mean here ensures a scalar for the feature vector.
                X_lung_combined(i, j) = mean(feature_val(:), 'omitnan'); 
            end
        end
        
        % Get label from the loaded master annotations table
        % This assumes lung_master_annotations_table has columns named 'filename' and 'quality'
        idx_anno = find(strcmp(lung_master_annotations_table.filename, current_filename_full), 1);

        if ~isempty(idx_anno)
            Y_lung_combined{i} = lung_master_annotations_table.quality{idx_anno};
        else
            % Fallback if a file found by dir() is NOT in your master annotations CSV
            Y_lung_combined{i} = 'UNLABELED_NO_MASTER_MATCH'; 
            fprintf('  Warning: Lung sound %s has no match in master annotations table. Marked as UNLABELED_NO_MASTER_MATCH.\n', current_filename_full);
        end

        fprintf('  Processed Lung Sound: %s, Features size: %s, Label: %s\n', current_filename_full, mat2str(size(X_lung_combined(i,:))), Y_lung_combined{i});
    end % End of for loop for lung files
    % Lung sound features
subplot(1,2,2);
features_to_plot = {'RMS', 'ZCR', 'SpectralCentroid'};
feature_data = [];
groups = [];
for i = 1:length(features_to_plot)
    feat_idx = find(strcmp(lung_feature_names, features_to_plot{i}));
    feature_data = [feature_data; X_lung_combined(:,feat_idx)];
    groups = [groups; i*ones(size(X_lung_combined,1),1)];
end
boxplot(feature_data, groups);
set(gca, 'XTickLabel', features_to_plot);
title('Lung Sound Feature Distribution');
ylabel('Normalized Value');
grid on;

saveas(gcf, fullfile(export_dir, 'feature_distributions.png'));
% Lung sounds
subplot(1,2,2);
lung_counts = countcats(categorical(Y_lung_combined));
pie(lung_counts);
title('Lung Sound Class Distribution');
legend(categories(categorical(Y_lung_combined)), 'Location', 'eastoutside');

saveas(gcf, fullfile(export_dir, 'class_distributions.png'));
% Pie charts of class distributions
figure('Position', [100, 100, 800, 400]);

% Heart sounds
subplot(1,2,1);
heart_counts = countcats(categorical(Y_heart_combined));
pie(heart_counts);
title('Heart Sound Class Distribution');
legend(categories(categorical(Y_heart_combined)), 'Location', 'eastoutside');


% Feature correlation heatmap
figure('Position', [100, 100, 1000, 800]);

% Heart features
subplot(1,2,1);
corr_matrix = corr(X_heart_combined, 'Rows', 'complete');
imagesc(corr_matrix);
colorbar;
title('Heart Sound Feature Correlation');
xticks(1:length(all_heart_feature_names));
yticks(1:length(all_heart_feature_names));
xticklabels(all_heart_feature_names);
yticklabels(all_heart_feature_names);
xtickangle(45);
set(gca, 'FontSize', 8);

% Lung features
subplot(1,2,2);
corr_matrix = corr(X_lung_combined, 'Rows', 'complete');
imagesc(corr_matrix);
colorbar;
title('Lung Sound Feature Correlation');
xticks(1:length(lung_feature_names));
yticks(1:length(lung_feature_names));
xticklabels(lung_feature_names);
yticklabels(lung_feature_names);
xtickangle(45);
set(gca, 'FontSize', 8);

saveas(gcf, fullfile(export_dir, 'feature_correlations.png'));



    % Remove rows with NaN features or 'UNLABELED' type labels
    valid_lung_rows = ~any(isnan(X_lung_combined), 2) & ...
                      ~strcmp(Y_lung_combined, 'UNLABELED_NO_FEATURES') & ...
                      ~strcmp(Y_lung_combined, 'UNLABELED_NO_MASTER_MATCH'); 

    num_valid_lung_samples = sum(valid_lung_rows);
    num_discarded_lung_samples = initial_lung_samples - num_valid_lung_samples;

    fprintf('Initial lung samples before filtering: %d\n', initial_lung_samples);
    fprintf('Number of lung samples discarded due to NaNs or UNLABELED types: %d\n', num_discarded_lung_samples);
    
    X_lung_combined = X_lung_combined(valid_lung_rows, :);
    Y_lung_combined = Y_lung_combined(valid_lung_rows);
    lung_file_lookup = lung_file_lookup(valid_lung_rows); % Update lookup as well

    fprintf('Consolidated Lung Feature matrix size: %s, Label vector size: %s\n', mat2str(size(X_lung_combined)), mat2str(size(Y_lung_combined)));
end % End of if-else block for empty features
% 1. Lung Sounds
lung_features_filepath = fullfile(export_dir, 'lung_features.csv');
lung_labels_filepath = fullfile(export_dir, 'lung_labels.csv');

writematrix(X_lung_combined, lung_features_filepath);
writecell(Y_lung_combined, lung_labels_filepath); % Use writecell for cell arrays of strings

fprintf('Lung features exported to: %s\n', lung_features_filepath);
fprintf('Lung labels exported to: %s\n', lung_labels_filepath);

% 2. Heart Sounds
heart_features_filepath = fullfile(export_dir, 'heart_features.csv');
heart_labels_filepath = fullfile(export_dir, 'heart_labels.csv');

writematrix(X_heart_combined, heart_features_filepath);
writecell(Y_heart_combined, heart_labels_filepath); % Use writecell for cell arrays of strings

fprintf('Heart features exported to: %s\n', heart_features_filepath);
fprintf('Heart labels exported to: %s\n', heart_labels_filepath);

fprintf('Export complete. You can now switch to Python for ML.\n');