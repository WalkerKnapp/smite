function [FitParams, FitParamsSE] = ...
    fitCDFOfJumps(MSDStruct, FitMethod, DiffusionModel, Verbose)
%fitCDFOfJumps fits the CDF of displacement data (jumps).
% This method will fit a model to the CDF (CPD) of trajectory
% displacements.
%
% INPUTS:
%   MSDStruct: A structure array of MSD data as output from computeMSD()
%              (see computeMSD() for more details). The MSDStruct must
%              already be populated with SortedJumps and CDFOfJumps (see
%              computeCDFOfMSD()).
%   FitMethod: A string specifying the fit method. (Default = 'WeightedLS')
%   DiffusionModel: A string specifying the diffusion model to fit to the
%                   MSD. See options in DiffusionEstimator class property
%                   'DiffusionModel'. (Default = 'Brownian2C')
%   Verbose: Verbosity level specifying how many temporary outputs should
%            be displayed (e.g., Command Window updates).
%
% OUTPUTS:
%   FitParams: Fit parameters for the MSD fit.  These will vary based on
%              'FitMethod'. FitParams(ii, :) will contain the fit
%              parameters for the fit to MSDStruct(ii). 
%              (numel(MSDStruct)xNParameters array)
%   FitParamsSE: Standard errors for the MSD fit parameters 'FitParams'.
%                These will vary based on 'Method'.

% Created by:
%   David J. Schodt (Lidke lab, 2021)


% Define default parameters if needed.
if (~exist('FitMethod', 'var') || isempty(FitMethod))
    FitMethod = 'WeightedLS';
end
if (~exist('DiffusionModel', 'var') || isempty(DiffusionModel))
    DiffusionModel = 'Brownian2C';
end
if (~exist('Verbose', 'var') || isempty(Verbose))
    Verbose = 0;
end

% Fit the MSD data.
if (Verbose > 1)
    fprintf(['fitCDFOfMSD(): fitting CDF of MSD with FitMethod = ', ...
        '''%s''...\n'], ...
        FitMethod)
end
NFits = numel(MSDStruct);
FitParams = NaN(NFits, 1);
FitParamsSE = NaN(NFits, 1);
for ii = 1:NFits
    % Isolate some arrays needed for the fit.
    CDFOfJumps = double(MSDStruct(ii).CDFOfJumps);
    FrameLags = double(MSDStruct(ii).FrameLags);
    NPoints = double(MSDStruct(ii).NPoints);
    LocVarianceSum = double(MSDStruct(ii).LocVarianceSum);
    SortedJumps = double(MSDStruct(ii).SortedJumps);

    % Fit the CDF of the jumps to the desired diffusion model.
    switch lower(DiffusionModel)
        case 'brownian1c'
            NComponents = 1;
        case 'brownian2c'
            NComponents = 2;
        otherwise
            error('Unknown ''DiffusionModel'' = %s', DiffusionModel)
    end
    [ParamsHat, ParamsHatSE] = ...
        smi_stat.DiffusionEstimator.fitCDFOfJumpsBrownian(...
        SortedJumps, CDFOfJumps, ...
        FrameLags, NPoints, LocVarianceSum, NComponents, ...
        [], FitMethod);
    FitParams(ii, :) = ParamsHat.';
    FitParamsSE(ii, :) = ParamsHatSE.';
end


end