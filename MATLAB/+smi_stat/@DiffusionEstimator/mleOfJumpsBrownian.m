function [MLEParams, MLEParamsSE] =  mleOfJumpsBrownian(...
    SquaredDisplacement, FrameLagsAll, LocVarianceSum, NComponents, ...
    FitOptions)
%mleOfJumpsBrownian finds the MLE for Brownian motion jumps.
% This method will estimate the MLE for the likelihood of observed squared
% displacements based on a Brownian motion model with either one diffusing
% population (i.e., one diffusion constant) or two diffusing populations
% (two diffusion constants and the population ratio).
%
% INPUTS:
%   SquaredDisplacement: The squared displacements made by the
%                        trajectory(ies). (NDatax1 numeric array)
%   FrameLagsAll: All of the frame lags associated with the jumps in
%                 'SquaredDisplacement'. (NDatax1 array)
%   LocVarianceSum: Sum of the localization variances for the two points
%                   used to compute the jumps. This array should be
%                   averaged over x and y.
%                   (NDatax1 numeric array)
%                   NOTE: I don't know which is better: average the
%                         variances, or average the SEs and square them? My
%                         bet is on averaging variances, but I'm not sure.
%                         This can make a big difference in some cases!
%   NComponents: Number of diffusion coefficients to fit.
%                (scalar, integer)(Default = 2)
%   FitOptions: Fit options sent directly to fminsearch (see doc fminsearch
%               for details)(Default = optimset(@fminsearch)
%               or optimoptions('fmincon') as appropriate)
%
% OUTPUTS:
%   MLEParams: MLE for the desired model parameters..  These will vary
%              based on 'FitMethod'. MLEParams(ii, :) will contain the MLEs
%              corresponding to the data in MSDStruct(ii).
%              (numel(MSDStruct)xNParameters array)
%   MLEParamsSE: Standard errors for the MLEs in 'MLEParams'. These will
%                vary based on 'Method'.
%
% REQUIRES:
%   Optimization Toolbox (for fmincon() when using the N-component models)

% Created by:
%   David J. Schodt (Lidke lab, 2021)


% Define default parameters if needed.
if (~exist('NComponents', 'var') || isempty(NComponents))
    NComponents = 2;
end
if (~exist('FitOptions', 'var') || isempty(FitOptions))
    if (NComponents > 1)
        FitOptions = optimoptions('fmincon');
    else
        FitOptions = optimset(@fminsearch);
    end
end

% Find the MLE based on the Brownian motion model.  If needed, compute the
% standard errors of parameters as the CRLB (which isn't correct, since our
% parameters are actually constrained!)
% NOTE: This model can be found by taking the prob(r|sigma^2=2Dt+loc.error)
%       (which is a product of Gaussians), converting to polar
%       coordinates, and integrating over theta. For multiple frame lags
%       (which we have), we'll also need to integrate over the frame lags
%       times the proportion of each frame lag observed.
CostFunction = @(Params) ...
    -smi_stat.DiffusionEstimator.brownianJumpLikelihood(Params, ...
    SquaredDisplacement, FrameLagsAll, LocVarianceSum);
ParamsInit = [0.1*ones(NComponents, 1); ...
    (1/NComponents)*ones(NComponents-1, 1)];
if (NComponents > 1)
    % Define lower and upper bounds for the fit parameters.
    ParamsLowerBound = zeros(2*NComponents - 1, 1);
    ParamsUpperBound = [inf(NComponents, 1); ...
        ones(NComponents-1, 1)];
    
    % Perform the constrained fit (for multiple components, we often need
    % to constrain the parameters to get the right answer).
    MLEParams = fmincon(CostFunction, ...
        ParamsInit, [], [], [], [], ...
        ParamsLowerBound, ParamsUpperBound);
else
    % For the single component fit, we'll just use fminsearch().
    MLEParams = fminsearch(CostFunction, ParamsInit, FitOptions);
end
if (nargout > 1)
    % Errors were requested for our MLE, so we'll return those as the CRLB
    % (which isn't correct, since we expect our parameters to be
    % constrained [at the least to be positive values]).
    DeltaHFraction = 1e-9;
    DeltaHBound = [1e-9; 1e-1];
    HessianAboutMLE = smi_stat.computeHessian(CostFunction, MLEParams, ...
        DeltaHFraction, DeltaHBound);
    MLEParamsSE = sqrt(diag(inv(HessianAboutMLE)));
end


end