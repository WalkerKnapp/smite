function [TRArray, FileList] = findDimerCandidatesFromFiles(...
    FileList1, FileList2, ...
    MaxDimerSeparation, MaxSeparation, MinValidPoints, MinMedianPhotons, ...
    BorderPadding, Verbose)
%findDimerCandidatesFromFiles creates a TRArray from the provided files.
% This method is a wrapper around findDimerCandidates() which will load
% Tracking Results (TR) structures from the provided files and construct
% the TRArray.
%
% INPUTS:
%   FileList1: Cell array containing the channel 1 file list, with the
%              entries each being a full path to a "*Results.mat" file
%              corresponding to channel 1 data saved in a 'TR' structure.
%   FileList2: Cell array containing the channel 2 file list, with the
%              ordering matching that of FileList1 (i.e., FileList2{ii} 
%              contains the channel 2 file corresponding to the channel 1
%              file FileList1{ii}).
%   MaxDimerSeparation: Maximum value of the minimum trajectory separation
%                       for a pair to be considered a dimer candidate.
%                       (pixels)(Default = 1)
%   MaxSeparation: Maximum distance between candidate trajectories that
%                  will be saved in the output TRArray.  For example, if
%                  two trajectories were separated by 
%                  [10, 5, 3, 1, 1, 4, 12], and the MaxSeparation is 4, the
%                  output TRArray will contain the segments of these
%                  trajectories corresponding to the separations 
%                  [3, 1, 1, 4]. 
%                  (pixels)(Default = 10 * MaxDimerDistance).
%   MinValidPoints: Minimum number of points during which the candidates
%                   must be within MaxSeparation of each other.
%                   (frames)(Default = 0)
%   MinPhotons: Minimum value for the median photons a trajectory must have
%               to be considered a candidate. (photons)(Default = 0)
%   BorderPadding: Minimum number of pixels from the border the
%                  trajectories must be during their candidate duration.
%                  (pixels)(Default = 0)
%   Verbose: Verbosity level of warning messages. (Default = 1)
%
% OUTPUTS:
%   TRArray: A structure array of TR structures, where the constituent TR
%            structures correspond to dimer candidate trajectories.  The 
%            first index corresponds to the "channel" of the trajectory and 
%            the second index corresponds to the pair number.  For example,
%            TRArray(j, 1) will contain information about a trajectory from 
%            TR1 that was observed within MaxDimerDistance of
%            TRArray(j, 2), a trajectory in TR2.
%            NOTE: The entirety of the two trajectories will be saved in
%                  the TRArray, and a new field DimerCandidateBool will be
%                  added to describe which datapoints correspond to the
%                  trajectories being close to each other (e.g., 
%                  TRArray(j, 1).FrameNum(TRArray(j, 1).DimerCandidateBool)
%                  will give the frames during which the trajectories were
%                  close).  Another field, ObservationStatus is defined as
%                  followed: ObservationStatus(1) is a boolean indicating
%                  whether or not the "off" to "on" transition
%                  (>MaxDimerSeparation to <=MaxDimerSeparation) was
%                  observed. ObservationStatus(2) is a boolean indicating
%                  whether or not the "on" to "off" transition was
%                  observed.
%   FileList: List of files sharing the same size as TRArray, such that
%             TRArray(ii, jj) is a trajectory present in FileList{ii, jj}.

% Created by:
%   David J. Schodt (Lidke Lab, 2020)


% Set default parameters if needed.
if (~exist('MaxDimerSeparation', 'var') || isempty(MaxDimerSeparation))
    MaxDimerSeparation = 1;
end
if (~exist('MaxSeparation', 'var') || isempty(MaxSeparation))
    MaxSeparation = 10 * MaxDimerSeparation;
end
if (~exist('MinValidPoints', 'var') || isempty(MinValidPoints))
    MinValidPoints = 0;
end
if (~exist('MinPhotons', 'var') || isempty(MinMedianPhotons))
    MinMedianPhotons = 0;
end
if (~exist('BorderPadding', 'var') || isempty(BorderPadding))
    BorderPadding = 0;
end
if (~exist('Verbose', 'var') || isempty(Verbose))
    Verbose = 0;
end

% Loop through the provided files, load them, and construct the TRArray.
FileList = {};
TRArray = struct([]);
for ff = 1:numel(FileList1)
    % Load the TR structures.
    try
        load(FileList1{ff}, 'TR')
        TR1 = TR;
    catch
        if Verbose
            warning(['findDimerCandidatesFromFiles(): ''TR'' structure ', ...
                'not found in file %s', FileList1{ff}])
        end
        continue
    end
    try
        load(FileList2{ff}, 'TR')
        TR2 = TR;
    catch
        if Verbose
            warning(['findDimerCandidatesFromFiles(): ''TR'' structure ', ...
                'not found in file %s', FileList2{ff}])
        end
        continue
    end
    
    % Find dimer candidates between 'TR1' and 'TR2'.
    TRArrayCurrent = smi_stat.HMM.findDimerCandidates(TR1, TR2, ...
        MaxDimerSeparation, MaxSeparation, ...
        MinValidPoints, MinMedianPhotons, BorderPadding);
    TRArray = smi_core.TrackingResults.catTR(TRArray, TRArrayCurrent);
    FileList = [FileList; ...
        repmat([FileList1(ff), FileList2(ff)], size(TRArrayCurrent, 1), 1)];
end


end