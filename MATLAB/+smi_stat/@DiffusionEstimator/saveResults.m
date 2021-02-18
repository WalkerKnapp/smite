function saveResults(obj)
%saveResults saves useful results of diffusion estimation analysis.
% This method can be used to save several pieces of information from the
% diffusion analysis results.

% Created by:
%   David J. Schodt (Lidke Lab, 2021)
    

% Define the path to the .mat file in which results will be saved.
if isempty(obj.SaveName)
    TimeString = smi_helpers.genTimeString('_');
    obj.SaveName = ['DiffusionResults_', TimeString, '.mat'];
end
if ~exist(obj.SaveDir, 'dir')
    mkdir(obj.SaveDir);
end

% Save some class properties to FileName.
DiffusionStruct = obj.DiffusionStruct;
FitMethod = obj.FitMethod;
MSDEnsemble = obj.MSDEnsemble;
MSDSingleTraj = obj.MSDSingleTraj;
MaxFrameLag = obj.MaxFrameLag;
save(fullfile(obj.SaveDir, obj.SaveName), ...
    'DiffusionStruct', 'FitMethod', ...
    'MSDEnsemble', 'MSDSingleTraj', 'MaxFrameLag');


end