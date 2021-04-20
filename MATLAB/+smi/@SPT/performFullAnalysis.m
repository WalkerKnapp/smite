function [TR, SMD] = performFullAnalysis(obj)
%performFullAnalysis fits and tracks data pointed to by obj.SMF
% This method is the main run method for the smi.SPT class, meaning that it
% will load raw data, perform gain/offset correction, fit localizations to
% the data, create trajectories from the localizations, and then save the
% results.
%
% OUTPUTS:
%   TR: Tracking Results structure (see smi_core.TrackingResults)
%   SMD: Single Molecule Data structure (see smi_core.SingleMoleculeData)

% Created by:
%   David J. Schodt (Lidke Lab, 2021)


% Prepare an SMLM class (we'll use this to load data, perform gain/offset
% correction, and fit the data).
SMLM = smi.SMLM(obj.SMF);
SMLM.Verbose = obj.Verbose;

% Load data, perform gain/offset correction, and fit the data.
SMLM.analyzeAll();
obj.SMD = SMLM.SMD;
obj.SMDPreThresh = SMLM.SMDPreThresh;

% Store the framerate and pixel size in SMD (if not already there).
if isempty(obj.SMD.FrameRate)
    obj.SMD.FrameRate = obj.SMF.Data.FrameRate;
end
if isempty(obj.SMD.PixelSize)
    obj.SMD.PixelSize = obj.SMF.Data.PixelSize;
end

% Generate trajectories from the localizations in obj.SMD.
[TR, SMD] = obj.generateTrajectories();


end