% Script to produce BaGoL results from SMITE *_Results.mat files.  The BaGoL
% results are placed in the subdirectory Results_BaGoL, assigned a name below,
% under the directory containing the *_Results.mat files.  BaGoLHier_analysis
% is called separately on each file in a parfor loop, assigning a worker for
% each dataset, so conducive to be run on a multi-core machine.
%
% See:
%
% Sub-Nanometer Precision using Bayesian Grouping of Localizations
% Mohamadreza Fazel, Michael J. Wester, David J. Schodt, Sebastian Restrepo
% Cruz, Sebastian Strauss, Florian Schueder, Thomas Schlichthaerle, Jennifer M.
% Gillette, Diane S. Lidke, Bernd Rieger, Ralf Jungmann, Keith A. Lidke
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

% ----------------------------------------------------------------------

% Output directory name.
%Results_BaGoL = 'Results_BaGoLHier';
Results_BaGoL = 'Xi=50_1';

% Generic parameters.
BaGoLParams.ImageSize = 256;        % (pixel)
%BaGoLParams.PixelSize = 108.018;    % (nm) [TIRF]
BaGoLParams.PixelSize = 97.8;       % (nm) [sequential]
BaGoLParams.OutputPixelSize = 4;    %2; % pixel size for posterior images (nm)
BaGoLParams.N_Burnin = 8000;        % Length of Burn-in chain
BaGoLParams.N_Trials = 2000;        % Length of post-burn-in chain
%BaGoLParams.N_Burnin = 20000;       % Length of Burn-in chain
%BaGoLParams.N_Trials = 10000;       % Length of post-burn-in chain
BaGoLParams.NSamples = 10;          % Number of samples before sampling Xi
BaGoLParams.N_NN =  5;              % Minimum number of nearest neighbors
                                    % required in filtering step

% Y_Adjust is sometimes needed to deal with lower left versus upper left
% y-origin issues.  Lower left with y increasing upwards is the default,
% requiring no changes, while upper left with y increasing downwards can
% sometimes occur, so Y must be replaced by Y - Y_Adjust, where Y_Adjust is the
% image size (see below) [pixels].
%BaGoLParams.Y_Adjust = BaGoLParams.ImageSize;
BaGoLParams.Y_Adjust = [];

% If Xi is a non-empty cell array, override the BaGoL.Xi value defined
% below, passing the corresponding (indexed by dataset) value to
% BaGoL_analysis.  This is useful for batch analysis when trying to make use of
% as many computer cores as possible in a single run.
Xi = {};
% Similar for DataROI, which should be dimensioned n_files x 4 if non-empty.
DataROI = [];

% SE_Adjust adds to X_SE and Y_SE, so inflates the precision.  For DNA_PAINT
% data, SE_Adjust = 1--2 nm, while for dSTORM, slightly bigger values should be
% used.
%
% k and theta below are the shape and scale parameters for the Gamma
% probability distribution function.  If just one parameter is provided,
% a Poisson distribution is used.
%
% DataROI is defined when running BaGoL's over only part of the
% image.  If DataROI is empty, use the whole image.
% 
% Note for batch runs, in which Files and Xi are input by hand, please see
% ### comments below.
% 
BaGoLParams.IntensityCutoff = 5000; % Intensity cutoff [DEPRECATED]
BaGoLParams.SE_Adjust = 3;          % Precision inflation applied to SE (nm)
BaGoLParams.ClusterDrift = 0;       % Expected magnitude of drift (nm/frame)
%BaGoLParams.ROIsz = 100;            % ROI size for RJMCMC (nm)
%BaGoLParams.OverLap = 25;           % Size of overlapping region (nm)
BaGoLParams.ROIsz = 500;            % ROI size for RJMCMC (nm)
BaGoLParams.OverLap = 50;           % Size of overlapping region (nm)
BaGoLParams.Xi = [50, 1];           % [k, theta] parameters for gamma prior
BaGoLParams.DataROI = [];           % [Xmin, Xmax, Ymin, Ymax] (pixel)

% File names and parameters per file.  If the parameters per file (Xi,
% DataROI) are not set below, use the generic values above.

%start_DataDir = '.';
%Files = uipickfiles('FilterSpec', start_DataDir, 'REFilter', ...
%                    '.*_ResultsStruct.*\.mat', 'Prompt',     ...
%                    '_ResultsStruct.mat files');
% ### Comment out the 3 lines above and use the commented out lines below when
% ### making batch runs, for example, on CARC.  Here, the files to process are
% ### defined relative to the directory where BaGoLHier_wrapper is run.
% ### Absolute pathnames are also fine, especially when used in conjunction
% ### with fullfile.
%'BaGoLHier/Data_2020-10-8-17-58-54_Results.mat'
%'BaGoLHier/SMD_DNA-Origami_MPI.mat'
%'BaGoLHier/SMR_dSTORM_EGFR.mat'
Files = {
'Data_2021-11-3-11-10-11_Results.mat'
};
%Xi = {
%[50, 1]
%};
% Define a single region of interest for each dataset (units are pixels).
% [YStart, XStart, YEnd, XEnd] = [163, 385, 233, 455]
% [Xmin, Xmax, Ymin, Ymax] (pixel)
% [385, 455, 163, 233]
% [412, 428, 190, 206]
DataROI = [
[120, 135, 120, 135]
];

% ----------------------------------------------------------------------

if iscell(Files)
   n_files = numel(Files);
else
   n_files = 0;
end
fprintf('Files to analyze = %d\n', n_files);
nXi = numel(Xi);
if nXi ~= 0 && nXi ~= n_files
   error('Xi must either be {} or contain %d values!', n_files);
end
if ~isempty(DataROI) && size(DataROI, 1) ~= n_files
   error('DataROI must either be [] or contain %d rows!', n_files);
end
% This code is needed to avoid DIPimage's split interfering with the parfor
% when using BaGoLParams.Xi rather than manually defining Xi above.
if nXi == 0
   Xi = cell(1, n_files);
   for i = 1 : n_files
      Xi{i} = BaGoLParams.Xi;
   end
end

if n_files > 0
   status = zeros(n_files, 1);

   % When using parfor, dispatch the dataset analyses to as many workers as
   % available.
   delete(gcp('nocreate'));
   MachineInfo = parcluster();
   NumWorkers = MachineInfo.NumWorkers;
   parpool('local', min(NumWorkers, n_files));

   % For debugging, it can be helpful to use the "for" rather than the
   % "parfor", and comment out the "try" line + "catch ... end" section below.
%  for i = 1 : n_files
   parfor i = 1 : n_files
      fprintf('(%d) %s ...\n', i, Files{i});
      [DataDir, File, Ext] = fileparts(Files{i});
      SaveDir = fullfile(DataDir, Results_BaGoL);

      % Set up Xi and DataROI for BGLParams.
      BGLParams = BaGoLParams;

      % Xi.
      BGLParams.Xi = Xi{i};
      if numel(BGLParams.Xi) == 1
         fprintf('Xi = %g\n', BGLParams.Xi);
      else
         fprintf('Xi = [%g, %g]\n', BGLParams.Xi);
      end

      % If DataROI is defined, override the default value given in BaGoLParams.
      if ~isempty(DataROI)
         BGLParams.DataROI = DataROI(i, :);
         fprintf('DataROI: [Xmin, Xmax, Ymin, Ymax] = [%g, %g, %g, %g]\n', ...
                 BGLParams.DataROI);
      end

      % Run BaGoLHier_analysis.
      try
         warning('OFF', 'stats:kmeans:FailedToConvergeRep');
         BaGoLHier_analysis(File, DataDir, SaveDir, BGLParams);
         status(i) = 1;
      catch ME
         fprintf('### PROBLEM with %s ###\n', Files{i});
         fprintf('%s\n', ME.identifier);
         fprintf('%s\n', ME.message);
         status(i) = -1;
      end
      fprintf('DONE (%d) %s.\n', i, Files{i});
      fprintf('\n');
   end

   fprintf('BaGoL status by file (-1 PROBLEM, 0 NOT DONE, 1 DONE):\n');
   for i = 1 : n_files
      fprintf('[%2d] %2d   %s\n', i, status(i), Files{i});
   end
end
warning('ON', 'stats:kmeans:FailedToConvergeRep');
fprintf('Done BaGoL.\n');
