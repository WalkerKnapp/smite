function processCell(obj, CellName)
%processCell will process data corresponding to CellName.
% This method will find the sub-directories for the cell CellName, which
% themselves contain the data for each label of the acquisition, and
% analyze that data.
%
% INPUTS:
%   CellName: Char. array/string of the Cell* directory (i.e., the distinct
%             Cell* names in the directory structure
%             *\Cell*\Label*\Data*.h5)

% Created by:
%   David J. Schodt (Lidke Lab 2021)


% Determine the names of the sub-directories of interest within
% CellName.  These correspond to single labels imaged during the
% super-resolution experiment.
LabelNames = smi_helpers.getDirectoryNames(...
    fullfile(obj.CoverslipDir, CellName), 'Label*');
NLabels = numel(LabelNames);
if (obj.Verbose > 1)
    fprintf('\tPublish.processCell(): %i label directories found:\n', ...
        NLabels)
    for ii = 1:NLabels
        fprintf('\t%s\n', LabelNames{ii})
    end
end

% Loop through each of the label directories and process the data.  If the
% processing fails on a given label ii, proceed with the next label anyways
% (these results might still be useful).
for ii = 1:NLabels
    % If LabelID was specified, skip all labels except those which exist in
    % LabelID.  However, if obj.LabelID is empty, then we wish to analyze
    % all LabelID's available.
    if ~(ismember(ii, obj.LabelID) || isempty(obj.LabelID))
        continue
    end
    
    % Attempt to process the data for label ii.
    try
        obj.processLabel(CellName, LabelNames{ii});
    catch MException
        if obj.Verbose
            warning(['Publish.processCell(): ', ...
                'Processing failed for %s\n%s, %s'], ...
                fullfile(CellName, LabelNames{ii}), ...
                MException.identifier, MException.message)
        end
    end
end

% If all labels for this cell were processed successfully, create an
% overlay image of the multiple labels, storing the overlay in the top
% level directory for easy access.
if (obj.GenerateSR && (NLabels>1))
    try
        obj.genSROverlays(...
            fullfile(obj.SaveBaseDir, CellName), ...
            obj.SaveBaseDir);
    catch MException
        if obj.Verbose
            warning(['Publish.processCell(): Overlay image ', ...
                'generation failed for %s\n%s, %s'], ...
                CellName, MException.identifier, MException.message)
        end
    end
end


end