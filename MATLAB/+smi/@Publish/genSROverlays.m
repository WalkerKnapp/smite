function [] = genSROverlays(ResultsCellDir, SaveDir)
%genSROverlays generates various types of  SR overlay images.
% This method will generate different types of overlay images for
% super-resolution (SR) data.  A multicolor overlay of the Gaussian SR
% images will be generated and saved as a .png file with the identifier
% _GaussianOverlay_ placed in the filename.  A multicolor overlay of the
% histogram SR images will be generated and saved as a .png file with the
% identifier _HistogramOverlay_ in the filename.  A multicolor overlay
% containing circles with radii proportional to the average localization
% precision (the generalized mean between the X and Y precisions) for each
% localization will be generated and saved as a .png file with the
% identifier _CircleOverlay_ in the filename.
%
% INPUTS:
%   ResultsCellDir: Directory containing the sub-directories for each label
%   SaveDir: Directory in which the resulting .png overlay images will be
%            saved.

% Created by:
%   David J. Schodt (Lidke Lab, 2018)


% Get the names of the directories containing the results for each label,
% throwing warnings if appropriate.
LabelDirNames = smi_helpers.getDirectoryNames(ResultsCellDir, 'Label*');
NLabels = numel(LabelDirNames);
if (NLabels > 4)
    warning('genSROverlay: 3 labels max')
    return
end
if (NLabels < 2)
    warning('genSROverlay: 2 labels min')
    return
end

% Load the Gaussian and histogram images for each label and store them in a
% single array.
% NOTE: pre-allocation assumes 5120x5120 images.
GaussianImages = zeros(5120, 5120, NLabels);
HistogramImages = zeros(5120, 5120, NLabels);
CircleImages = zeros(0, 0, 0, 'uint8'); % initialize
MinPixelsPerCircle = 16; % ~min. num. of pixels in each circle
BitDepth = 8; % specific to thesave type: this is for a 8 bit png
for ii = 1:NLabels
    % Create a list of sub-directories under the current label (there could
    % be multiple for a given label, e.g. an extra for a photobleaching
    % round of imaging).
    DatasetDirNames = SMA_Publish.getDirectoryNames(...
        fullfile(ResultsCellDir, LabelDirNames{ii}), 'Data*');
    
    % If more than two datasets exists for this label, throw an error (we
    % can have two: one desired result, one photobleaching result) since
    % it's not clear which dataset
    % to use.
    if numel(DatasetDirNames) > 2
        warning('More than two datasets exist for %s', ...
            LabelDirNames{ii})
    end
    
    % Load our images into the appropriate arrays.
    for jj = 1:numel(DatasetDirNames)
        % Ensure that we skip results from photobleaching rounds.
        if contains(DatasetDirNames{jj}, 'bleach', 'IgnoreCase', true)
            continue
        end
        
        % Create the appropriate filepaths and read in the images.
        FileDirectory = fullfile(ResultsCellDir, ...
            LabelDirNames{ii}, DatasetDirNames{jj});
        FileNameCircle = sprintf('%s_Results_CircleIm.png', ...
            DatasetDirNames{jj});
        FileNameGaussian = sprintf('%s_Results_GaussIm.png', ...
            DatasetDirNames{jj});
        FileNameHistogram = sprintf('%s_Results_HistIm.png', ...
            DatasetDirNames{jj});
        CircleImages(:, :, ii) = imread(...
            fullfile(FileDirectory, FileNameCircle));
        GaussianImages(:, :, ii) = imread(...
            fullfile(FileDirectory, FileNameGaussian));
        HistogramImages(:, :, ii) = imread(...
            fullfile(FileDirectory, FileNameHistogram));
    end
end

% Generate our color overlay images (3 channel images).
[OverlayImageGaussian, ColorOrderTagGaussian] = ...
    SMA_Publish.overlayNImages(GaussianImages);
[OverlayImageHistogram, ColorOrderTagHistogram] = ...
    SMA_Publish.overlayNImages(HistogramImages);
[OverlayImageCircle, ColorOrderTagCircle] = ...
    SMA_Publish.overlayNImages(CircleImages);

% Save the overlay images in the top level directory.
CellName = ResultsCellDir(regexp(ResultsCellDir, 'Cell*'):end);
CellNameClean = erase(CellName, '_'); % remove underscore(s)
OverlayImageGaussianName = sprintf('%s_GaussianOverlay_%s.png', ...
    CellNameClean, ColorOrderTagGaussian);
imwrite(OverlayImageGaussian, ...
    fullfile(SaveDir, OverlayImageGaussianName));
OverlayImageHistogramName = sprintf('%s_HistogramOverlay_%s.png', ...
    CellNameClean, ColorOrderTagHistogram);
imwrite(OverlayImageHistogram, ...
    fullfile(SaveDir, OverlayImageHistogramName));
OverlayImageCircleName = sprintf('%s_CircleOverlay_%s.png', ...
    CellNameClean, ColorOrderTagCircle);
imwrite(OverlayImageCircle, ...
    fullfile(SaveDir, OverlayImageCircleName));


end