
clc;
clear;
close all;

%% 1. Define Input and Output Folders

% Relative paths based on the MATLAB script location
scriptFolder = fileparts(mfilename('fullpath'));

folderPath = fullfile(scriptFolder, 'Raw Data');
folderWritePath = fullfile(scriptFolder, 'Filtered Data');

% Check input folder
if ~exist(folderPath, 'dir')
    error('Input folder not found: %s', folderPath);
end

% Create output folder if it does not exist
if ~exist(folderWritePath, 'dir')
    mkdir(folderWritePath);
end

% List all TXT files
files = dir(fullfile(folderPath, '*.txt'));
fileNames = {files.name};

fprintf('Total files found: %d\n', length(fileNames));

%% 2. Process Each Point Cloud

for fileIndex = 1:length(fileNames)

    tic;

    fprintf('\nProcessing file %d of %d: %s\n', ...
        fileIndex, length(fileNames), fileNames{fileIndex});

    %% 3. Read XYZ Point Cloud

    inputFile = fullfile(folderPath, fileNames{fileIndex});

    data = readmatrix(inputFile);

    PCL = data(:, 1:3);

    PC = pointCloud(PCL);

    fprintf('Input points: %d\n', size(PCL, 1));

    %% 4. Calculate Mean Point Spacing and Search Radius

    [~, D2] = knnsearch(PCL, PCL, 'K', 2);

    PS_DS = mean(D2(:, 2));

    radius = 10 * PS_DS - 16 * PS_DS^2;

    fprintf('Mean point spacing: %.4f m\n', PS_DS);
    fprintf('Search radius: %.4f m\n', radius);

    %% 5. Calculate Local Eigenvalues

    numPoints = size(PCL, 1);

    Lambda = NaN(numPoints, 3);

    parfor i = 1:numPoints

        point = PCL(i, :);

        [indices, ~] = findNeighborsInRadius(PC, point, radius);

        if numel(indices) > 10

            points = PCL(indices, :);

            [~, ~, latent] = pca(points);

            Lambda(i, :) = latent';

        end
    end

    % Remove points with invalid eigenvalues
    validIdx = ~isnan(Lambda(:, 1));

    PCL = PCL(validIdx, :);
    Lambda = Lambda(validIdx, :);

    fprintf('Valid points: %d\n', size(PCL, 1));

    %% 6. Calculate Surface Curvature

    E1 = Lambda(:, 1);
    E2 = Lambda(:, 2);
    E3 = Lambda(:, 3);

    curvature = E3 ./ (E1 + E2 + E3);

    %% 7. Calculate 90th Percentile Curvature Threshold

    percentile = 0.90;

    [counts, edges] = histcounts(curvature, ...
        'Normalization', 'cdf');

    binIndex = find(counts >= percentile, 1, 'first');

    if isempty(binIndex)
        error('Could not determine the 90th percentile.');
    end

    lowerEdge = edges(binIndex);
    upperEdge = edges(binIndex + 1);
    binCDF = counts(binIndex);

    if binIndex == 1
        lowerCDF = 0;
    else
        lowerCDF = counts(binIndex - 1);
    end

    threshold = lowerEdge + ...
        (percentile - lowerCDF) / (binCDF - lowerCDF) * ...
        (upperEdge - lowerEdge);

    fprintf('Curvature threshold: %.6f\n', threshold);

    %% 8. Extract Candidate Roof-Bolt Points

    tid = curvature > threshold;

    PC_bolts = PCL(tid, :);

    fprintf('Candidate bolt points: %d\n', size(PC_bolts, 1));

    %% 9. DBSCAN Clustering

    epsilon = 0.05;
    minpts = 10;

    [idx, ~] = dbscan(PC_bolts, epsilon, minpts);

    numClusters = max(idx);

    fprintf('DBSCAN clusters detected: %d\n', numClusters);

    %% 10. Cluster-Based Point Cloud Refinement

    PCL_filtered = [];

    for i = 1:numClusters

        clusterIdx = idx == i;

        bolt_cloud = PC_bolts(clusterIdx, :);

        if size(bolt_cloud, 1) > 400

            % Retain large clusters directly
            PCL_filtered = [PCL_filtered; bolt_cloud];

        else

            % Expand smaller clusters using nearby points
            centroid = mean(bolt_cloud, 1);

            searchRadius = 0.1;

            distances = sqrt(sum((PCL - centroid).^2, 2));

            pointsWithinRadius = PCL(distances <= searchRadius, :);

            PCL_filtered = [PCL_filtered; pointsWithinRadius];

        end
    end

    % Remove duplicate points
    PCL_filtered = unique(PCL_filtered, 'rows');

    fprintf('Final filtered points: %d\n', ...
        size(PCL_filtered, 1));

    %% 11. Save Filtered Point Cloud as TXT

    outputFile = fullfile(folderWritePath, fileNames{fileIndex});

    fileID = fopen(outputFile, 'w');

    if fileID == -1
        error('Could not open output file: %s', outputFile);
    end

    for i = 1:size(PCL_filtered, 1)

        for j = 1:3

            formatted_number = sprintf('%.4f', PCL_filtered(i, j));

            if j < 3
                fprintf(fileID, '%s ', formatted_number);
            else
                if i == size(PCL_filtered, 1)
                    fprintf(fileID, '%s', formatted_number);
                else
                    fprintf(fileID, '%s\n', formatted_number);
                end
            end
        end
    end

    fclose(fileID);

    fprintf('Saved: %s\n', outputFile);

    toc;

end

disp('All point clouds processed successfully.');
