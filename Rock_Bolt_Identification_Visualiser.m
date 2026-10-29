clc;
clear;
close all;

%% 1. Define Folder Paths

% Relative paths based on MATLAB script location
scriptFolder = fileparts(mfilename('fullpath'));

rawFolder = fullfile(scriptFolder, 'Raw Data');
filteredFolder = fullfile(scriptFolder, 'Filtered Data');
predictionFolder = fullfile(scriptFolder, 'Stage 2_Semantic Segmentation\log\sem_seg\pointnet2_sem_seg\visual');

% Output folder for both TXT and FIG files
outputFolder = fullfile(scriptFolder, 'Predicted Bolts');

% Create output folder if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Read all raw point cloud files
files = dir(fullfile(rawFolder, '*.txt'));

fprintf('Total files found: %d\n', length(files));


%% 2. Process Each Point Cloud

for fileIndex = 1:length(files)

    fileName = files(fileIndex).name;
    [~, baseName, ~] = fileparts(fileName);

    fprintf('\nProcessing file %d of %d: %s\n', ...
        fileIndex, length(files), fileName);

    %% 3. Read Raw and Curvature-Filtered Point Clouds

    rawFile = fullfile(rawFolder, fileName);
    filteredFile = fullfile(filteredFolder, fileName);
    predictionFile = fullfile(predictionFolder, fileName);

    % Check required files
    if ~isfile(filteredFile) || ~isfile(predictionFile)
        warning('Missing filtered cloud or predictions for %s', ...
            fileName);
        continue;
    end

    % Read XYZ coordinates
    rawData = readmatrix(rawFile);
    filteredData = readmatrix(filteredFile);

    rawXYZ = rawData(:, 1:3);
    filteredXYZ = filteredData(:, 1:3);

    %% 4. Read Semantic Segmentation Predictions

    labels = readmatrix(predictionFile);
    labels = labels(:);

    % Verify point-label correspondence
    if numel(labels) ~= size(filteredXYZ, 1)
        warning('Prediction count does not match filtered points: %s', ...
            fileName);
        continue;
    end

    % Check binary labels
    if any(~ismember(labels, [0, 1]))
        warning('Invalid prediction labels in %s', fileName);
        continue;
    end

    %% 5. Extract Predicted Roof-Bolt Points

    boltIdx = labels == 1;

    boltXYZ = filteredXYZ(boltIdx, :);

    fprintf('Raw points: %d\n', size(rawXYZ, 1));
    fprintf('Filtered points: %d\n', size(filteredXYZ, 1));
    fprintf('Predicted bolt points: %d\n', size(boltXYZ, 1));

    %% 6. Save Predicted Bolt Points as TXT

    outputTXT = fullfile(outputFolder, [baseName, '.txt']);

    fileID = fopen(outputTXT, 'w');

    if fileID == -1
        error('Could not open output file: %s', outputTXT);
    end

    for i = 1:size(boltXYZ, 1)

        fprintf(fileID, '%.4f %.4f %.4f', ...
            boltXYZ(i, 1), boltXYZ(i, 2), boltXYZ(i, 3));

        if i < size(boltXYZ, 1)
            fprintf(fileID, '\n');
        end

    end

    fclose(fileID);

    fprintf('Predicted bolt points saved: %s\n', outputTXT);

    %% 7. Visualise Predicted Bolts on Raw Point Cloud

    figure('Color', 'w', ...
        'Name', ['Predicted Roof Bolts - ', baseName]);

    % Raw point cloud in grey
    rawColor = repmat(uint8([160 160 160]), ...
        size(rawXYZ, 1), 1);

    pcshow(rawXYZ, rawColor, 'MarkerSize', 15);

    hold on;

    % Overlay predicted bolt points in red
    if ~isempty(boltXYZ)

        pcshow(boltXYZ, 'r', 'MarkerSize', 35);

    end

    title(['Predicted Roof Bolts - ', baseName], ...
        'Interpreter', 'none');

    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');

    axis equal;
    grid off;

    set(gcf, 'Color', 'w');
    set(gca, 'Color', 'w');

    view(3);
    rotate3d on;

end

disp('All point clouds processed and predicted bolt TXT files saved.');
