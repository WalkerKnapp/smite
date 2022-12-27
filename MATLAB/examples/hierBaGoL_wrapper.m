% Script to produce BaGoL results from SMITE *_Results.mat files.  The BaGoL
% results are placed in the subdirectory Results_BaGoL, assigned a name below,
% under the directory containing the *_Results.mat files.  hierBaGoL_analysis
% is called separately on each file in a parfor loop, assigning a worker for
% each dataset, so conducive to be run on a multi-core machine.
%
% The wrapper sets parameters and lists files (full paths) to be analyzed and
% optional ROIs to apply in the next to last section.
%
% For _Results.mat files with large numbers of localizations (> 300,000 or so),
% hierBaGoL may crash (or partially crash), so should not be part of a parfor
% loop as a crash will cause ALL of the non-finished parallel jobs to restart.
% Such _Results.mat files should be analyzed in separate MATLABs.
%
% NOTE: MAPN_*.mat files are always produced containng simply the MAPN
% coordinates.  See +smi/@BaGoL/hierBaGoL_analysis.m for more details on files
% produced.
%
% See:
%
% Sub-Nanometer Precision using Bayesian Grouping of Localizations
% Mohamadreza Fazel, Michael J. Wester, David J. Schodt, Sebastian Restrepo
% Cruz, Sebastian Strauss, Florian Schueder, Thomas Schlichthaerle, Jennifer M.
% Gillette, Diane S. Lidke, Bernd Rieger, Ralf Jungmann and Keith A. Lidke
% (Nature Communications, 2022)
% 
% Single-molecule localization microscopy super-resolution methods rely on
% stochastic blinking events, which can occur multiple times from each emitter
% over the course of data acquisition.  Typically the blinking events from each
% emitter are treated as independent events, without an attempt to assign them
% to a particular emitter.  We describe a Bayesian method of grouping and
% combining localizations from multiple blinking events that can improve
% localization precision and can provide better than one nanometer precision
% under achievable experimental conditions.  The statistical distribution of
% the number of blinking events per emitter along with the precision of each
% localization event are used to generate a posterior probability distribution
% of the number and location of emitters.  The blinks per emitter distribution
% can be input or learned from data.  We demonstrate the method on a range of
% synthetic data, DNA origami constructs and biological structures using
% DNA-PAINT and dSTORM.
%
% Pre-filtering actions (frame connection and NN not used for dSTORM data):
%
% SR data -> remove localizations with negative coordinates
%         -> intensity filter (InMeanMultiplier)
%         -> inflate standard errors (SE_Adjust)
%         -> frame connection, removing connections which involve only N_FC
%            frames
%         -> Nearest Neighbor filter (N_NN) --- Do not use on dSTORM data!
%         -> BaGoL (via parfor calling hierBaGoL_analysis on each dataset)
%
% The pre-filtering is all now in SMLM, although SE_Adjust can be set here as
% well if not done previously.

% ----------------------------------------------------------------------

% Output directory name.
%Results_BaGoL = 'Results_BaGoLHier';
Results_BaGoL = 'preC';

% Generic parameters.
BaGoLParams.ImageSize = 256;        % (pixel)
%BaGoLParams.PixelSize = 108.018;    % (nm) [TIRF]
BaGoLParams.PixelSize = 97.8;       % (nm) [sequential]
BaGoLParams.OutputPixelSize = 4;    %2; % pixel size for posterior images (nm)
BaGoLParams.N_Burnin = 32000;       % Length of Burn-in chain
BaGoLParams.N_Trials = 8000;        % Length of post-burn-in chain
%BaGoLParams.N_Burnin = 8000;        % Length of Burn-in chain
%BaGoLParams.N_Trials = 2000;        % Length of post-burn-in chain
BaGoLParams.NSamples = 10;          % Number of samples before sampling Xi
BaGoLParams.ClusterDrift = 0;       % Expected magnitude of drift (nm/frame)

% Y_Adjust is sometimes needed to deal with lower left versus upper left
% y-origin issues.  Lower left with y increasing upwards is the default,
% requiring no changes, while upper left with y increasing downwards can
% sometimes occur, so Y must be replaced by Y - Y_Adjust, where Y_Adjust is the
% image size (see below) [pixels].
%BaGoLParams.Y_Adjust = BaGoLParams.ImageSize;
BaGoLParams.Y_Adjust = [];

% SE_Adjust adds to X_SE and Y_SE, so inflates the precision.  For DNA_PAINT
% data, SE_Adjust = 1--2 nm, while for dSTORM, slightly bigger values should
% be used.  Note that this quantity can be specified as an array of length
% n_files if applied differently to each file.
BaGoLParams.SE_Adjust = 0;          % Precision inflation applied to SE (nm)
%BaGoLParams.SE_Adjust = [0, 0];     % Precision inflation applied to SE (nm)

% The values for ROIsz and OverLap directly below are good for denser data as
% less computational effort is required, so the code runs faster.  The second
% set of values can be used for sparser data to generate larger ROIs, but may
% produce artifacts with dense data.
BaGoLParams.ROIsz = 500;            % ROI size for RJMCMC (nm)
BaGoLParams.OverLap = 50;           % Size of overlapping region (nm)
BaGoLParams.Cutoff = 30;            % Pre-clustering cutoff (nm)
%BaGoLParams.ROIsz = 50;             % ROI size for RJMCMC (nm)
%BaGoLParams.OverLap = 10;           % Size of overlapping region (nm)
%BaGoLParams.ROIsz = 100;            % ROI size for RJMCMC (nm)
%BaGoLParams.OverLap = 25;           % Size of overlapping region (nm)
%BaGoLParams.ROIsz = 500;            % ROI size for RJMCMC (nm)
%BaGoLParams.OverLap = 50;           % Size of overlapping region (nm)

% k and theta below are the shape and scale parameters for the Gamma
% probability distribution function.  If just one parameter is provided,
% a Poisson distribution is used.
BaGoLParams.Xi = [20, 1];           % [k, theta] parameters for gamma prior

% Note for batch runs, in which Files and DataROI are input by hand, please see
% ### comments below.
BaGoLParams.DataROI = [];           % [Xmin, Xmax, Ymin, Ymax] (pixel)
DataROI = [];

% If ROIs is true, the input file has ROIs already defined (*_ROIs.txt), so use
% them below.
ROIs = false;
ROIs = true;

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

%start_DataDir = '.';
%Files = uipickfiles('FilterSpec', start_DataDir, 'REFilter', ...
%                    '.*_ResultsStruct.*\.mat', 'Prompt',     ...
%                    '_ResultsStruct.mat files');
% ### Comment out the 4 lines above and use the commented out lines below when
% ### making batch runs, for example, on CARC.  Here, the files to process are
% ### defined relative to the directory where hierBaGoL_wrapper is run.
% ### Absolute pathnames are also fine, especially when used in conjunction
% ### with fullfile.
%'BaGoLHier/Data_2020-10-8-17-58-54_Results.mat'
D1 = '../DATA';
Files = {
fullfile(D1, 'Cell_02_Label_01_Results.mat');
};
%fullfile(D1, 'Cell_03_Label_01_Results.mat');

% DataROI is defined when running BaGoL over only part of the image.
% If DataROI is empty, use the whole image.
% 
% Define a single region of interest for each dataset (units are pixels).
% [YStart, XStart, YEnd, XEnd] = [163, 385, 233, 455]
% [Xmin, Xmax, Ymin, Ymax] (pixel)
% [385, 455, 163, 233]
%DataROI = [
%[120, 136, 190, 206]
%[110, 126,  90, 106]
%];

if numel(Files) == 1 && ROIs
   [DataDir, File, Ext] = fileparts(Files{1});
   basename = fullfile(DataDir, 'Analysis', File);
   % Assume SMD files are of the form Cell_nn_Label_0n_Results.mat and RoI
   % files are of the form Cell_nn_Label_01_Results_ROIs.mat
   filename = regexprep(basename, 'Label_02', 'Label_01');

   ROIsFile = load([filename, '_ROIs.mat']);
   n_ROIs = numel(ROIsFile.RoI);
   DataROI = zeros(n_ROIs, 4);
   for i = 1 : n_ROIs
      DataROI(i, :) = ROIsFile.RoI{i}.ROI ./ BaGoLParams.PixelSize;
   end

   Files = cell(n_ROIs, 1);
   for i = 1 : n_ROIs
      Files{i} = sprintf('%s_ROI_%02d.mat', basename, i);
   end

end

% ----------------------------------------------------------------------

% Run the BaGoL analyses.
smi.BaGoL.hierBaGoL_run(Files, DataROI, Results_BaGoL, BaGoLParams, ROIs);

fprintf('Done BaGoL.\n');
