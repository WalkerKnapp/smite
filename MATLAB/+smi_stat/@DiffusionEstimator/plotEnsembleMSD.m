function [PlotAxes] = plotEnsembleMSD(PlotAxes, ...
    MSDEnsemble, DiffusionStruct, DiffusionModel, UnitFlag)
%plotEnsembleMSD plots an ensemble MSD and an associated fit.
% This method will plot the MSD data in MSDEnsemble as well as the fit
% information provided in DiffusionStruct.
%
% INPUTS:
%   PlotAxes: Axes in which the plot will be made. (Default = gca())
%   MSDEnsemble: Structure array with ensemble MSD results (see outputs of
%                computeMSD())
%   DiffusionStruct: Structure array containing MSD fit ressults. 
%                    (Default = [], meaning no fit results are plotted).
%   DiffusionModel: A string specifying the diffusion model to fit to the
%                   MSD. See options in DiffusionEstimator class property
%                   'DiffusionModel'. (Default = 'Brownian')
%   UnitFlag: Flag to specify camera units (0) or physical units (1).
%             (Default = 0)
%
% OUTPUTS:
%   PlotAxes: Axes in which the plot was made.

% Created by:
%   David J. Schodt (Lidke lab, 2021)


% Set defaults if needed.
if (~exist('PlotAxes', 'var') || isempty(PlotAxes))
    PlotAxes = gca();
end
if (~exist('DiffusionStruct', 'var') || isempty(DiffusionStruct))
    DiffusionStruct = [];
end
if (~exist('DiffusionModel', 'var') || isempty(DiffusionModel))
    DiffusionModel = 'Brownian';
end
if (~exist('UnitFlag', 'var') || isempty(UnitFlag))
    UnitFlag = 0;
end

% Plot the MSD.
FrameConversion = UnitFlag/DiffusionStruct(2).FrameRate + ~UnitFlag;
MSDConversion = ~UnitFlag ...
    + UnitFlag*(DiffusionStruct(2).PixelSize^2);
plot(PlotAxes, MSDEnsemble.FrameLags*FrameConversion, ...
    MSDEnsemble.MSD*MSDConversion, '.')
hold(PlotAxes, 'on')

% If needed, plot the fit results.
if ~isempty(DiffusionStruct)
    switch DiffusionModel
        case {'Brownian', 'brownian'}
            % The Brownian diffusion model suggests the MSD is linear with
            % time.
            FitParams = DiffusionStruct(2).FitParams;
            FrameArray = ...
                MSDEnsemble.FrameLags([1, numel(MSDEnsemble.FrameLags)]);
            plot(PlotAxes, FrameArray*FrameConversion, ...
                MSDConversion * (FitParams(2)*FrameArray + FitParams(1)))
    end
end
TimeUnit = smi_helpers.stringMUX({'frames', 'seconds'}, UnitFlag);
MSDUnit = smi_helpers.stringMUX(...
    {'pixels^2', 'micrometers^2'}, UnitFlag);
xlabel(PlotAxes, TimeUnit)
ylabel(PlotAxes, MSDUnit)
legend(PlotAxes, {'MSD', 'Fit'}, 'Location', 'best')


end