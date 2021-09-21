function gui(obj, GUIParent)
%gui is the GUI method for the ChannelRegistration class.
% This method generates a GUI for the ChannelRegistration class which
% allows the user to interactively load fiducials and generate transforms.
%
% INPUTS:
%   GUIParent: The 'Parent' of this GUI, e.g., a figure handle.
%              (Default = figure(...))

% Created by:
%   David J. Schodt (Lidke lab, 2020)


% Create a figure handle for the GUI if needed.
if ~(exist('GUIParent', 'var') && ~isempty(GUIParent) ...
        && isgraphics(GUIParent))
    DefaultFigurePosition = get(0, 'defaultFigurePosition');
    GUIParent = figure('MenuBar', 'none', ...
        'Name', 'Channel Registration', 'NumberTitle', 'off', ...
        'Units', 'pixels', ...
        'Position', DefaultFigurePosition);
end

% Generate some panels to help organize the GUI.
FilePanel = uipanel(GUIParent, 'Title', 'Fiducial files', ...
    'Units', 'normalized', 'Position', [0, 0.5, 0.5, 0.5]);
FittingPanel = uipanel(GUIParent, 'Title', 'Fiducial fitting', ...
    'Units', 'normalized', 'Position', [0, 0, 0.5, 0.5]);
TransformParametersPanel = uipanel(GUIParent, ...
    'Title', 'Transform parameters', ...
    'Units', 'normalized', 'Position', [0.5, 0.5, 0.5, 0.5]);
TransformControlsPanel = uipanel(GUIParent, ...
    'Title', 'Transform controls', ...
    'Units', 'normalized', 'Position', [0.5, 0, 0.5, 0.5]);

% Add some controls to the fiducial file panel.
ButtonSize = [0, 0, 1, 0.12];
TextSize = [0, 0, 0.5, 0.12];
PopupSize = [0, 0, 1, 0.5];
EditSize = [0, 0, 0.5, 0.12];
ControlHandles.AddFileButton = uicontrol(FilePanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, ...
    'String', 'Add fiducial(s)', ...
    'Units', 'normalized', ...
    'Tooltip', 'Select fiducial image files', ...
    'Position', ButtonSize+[0, 1-ButtonSize(4), 0, 0], ...
    'Callback', @addFile);
ControlHandles.RemoveFileButton = uicontrol(FilePanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, ...
    'String', 'Remove selected fiducial', ...
    'Units', 'normalized', ...
    'Tooltip', 'Remove fiducial image file', ...
    'Position', ButtonSize+[0, 1-2*ButtonSize(4), 0, 0], ...
    'Callback', @removeFile);
FiducialFileTooltip = ...
    sprintf(['Fiducial filenames. The currently selected file in\n', ...
    'in the list will be used as the reference.']);
ControlHandles.FileList = uicontrol(FilePanel, ...
    'Style', 'listbox', ...
    'Units', 'normalized', ...
    'Position', PopupSize+[0, 1-2*ButtonSize(4)-PopupSize(4), 0, 0], ...
    'Tooltip', FiducialFileTooltip, ...
    'Callback', @selectReference);
ControlHandles.ReferenceFileText = uicontrol(FilePanel, ...
    'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Reference fiducial: ', ...
    'Visible', 'off', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', FiducialFileTooltip, ...
    'Units', 'normalized', ...
    'Position', ...
    TextSize + [0, 1-2*ButtonSize(4)-PopupSize(4)-TextSize(4), 0, 0]);
ControlHandles.ReferenceFileName = uicontrol(FilePanel, ...
    'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'Visible', 'off', ...
    'HorizontalAlignment', 'left', ...
    'Tooltip', FiducialFileTooltip, ...
    'Units', 'normalized', 'Position', TextSize ...
    + [TextSize(3), 1-2*ButtonSize(4)-PopupSize(4)-TextSize(4), 0, 0]);
ROISplittingTooltip = sprintf(...
    ['If your fiducial file contains multiple fiducials in the\n', ...
    'same image (i.e., different ROIs of the image), you can define\n', ...
    'the way the image is split in this field.  This field becomes\n', ...
    'the class property ''SplitFormat'' (see \n', ...
    'convertSplitFormatToROIs() for more details)']);
ControlHandles.ROISplittingText = uicontrol(FilePanel, ...
    'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'ROI splitting format: ', ...
    'Visible', 'off', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', ROISplittingTooltip, ...
    'Units', 'normalized', ...
    'Position', ...
    TextSize + [0, 1-2*ButtonSize(4)-PopupSize(4)-TextSize(4), 0, 0]);
ControlHandles.ROISplittingPopup = uicontrol(FilePanel, ...
    'Style', 'popup', ...
    'String', obj.SplitFormatOptionsChar, ...
    'FontUnits', 'normalized', 'FontSize', 0.5, ...
    'Visible', 'off', ...
    'HorizontalAlignment', 'center', ...
    'Tooltip', ROISplittingTooltip, ...
    'Units', 'normalized', 'Position', TextSize ...
    + [TextSize(3), 1-2*ButtonSize(4)-PopupSize(4)-TextSize(4), 0, 0], ...
    'Callback', @refresh);

% Add some controls to the fitting panel.
ButtonSize = [0, 0, 1, 0.12];
TextSize = [0, 0, 0.5, 0.12];
CheckSize = [0, 0, 0.5, 0.12];
ControlHandles.SMFButton = uicontrol(FittingPanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Modify fitting parameters', ...
    'Units', 'normalized', ...
    'Position', ButtonSize+[0, 1-ButtonSize(4), 0, 0], ...
    'Tooltip', sprintf(...
    ['Open SMF GUI to adjust the fiducial fitting parameters.\n', ...
    'Note that this is only relevant for coordinate transforms\n', ...
    '(i.e., when ChannelRegistration.TransformBasis = ''coords'').']), ...
    'Callback', @openSMFGUI);
CullTooltipString = sprintf(...
    ['Selecting manual cull will allow you to cull\n', ...
    'fiducial localizations that you don''t want to include when\n', ...
    'computing the transform.']);
uicontrol(FittingPanel, 'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Manual cull: ', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', CullTooltipString, ...
    'Units', 'normalized', ...
    'Position', TextSize+[0, 1-ButtonSize(4)-TextSize(4), 0, 0]);
ControlHandles.CullCheckbox = uicontrol(FittingPanel, ...
    'Style', 'checkbox', 'Value', obj.ManualCull, ...
    'Units', 'normalized', ...
    'Position', CheckSize ...
    + [TextSize(3), 1-ButtonSize(4)-TextSize(4), 0, 0], ...
    'Tooltip', CullTooltipString, ...
    'Callback', @refresh);
AutoscaleTooltip = ...
    sprintf(['Autoscale fiducial images before fitting.\n', ...
    'Selecting this will use an empirical autoscaling on the\n', ...
    'fiducial images, which generally seems to help the fitting.\n', ...
    'This parameter only matters for coordinate type transforms']);
uicontrol(FittingPanel, 'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Autoscale fiducials: ', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', AutoscaleTooltip, ...
    'Units', 'normalized', ...
    'Position', TextSize+[0, 1-ButtonSize(4)-2*TextSize(4), 0, 0]);
ControlHandles.AutoscaleFiducialCheckbox = uicontrol(FittingPanel, ...
    'Style', 'checkbox', 'Value', obj.AutoscaleFiducials, ...
    'Units', 'normalized', ...
    'Position', CheckSize ...
    + [TextSize(3), 1-ButtonSize(4)-2*TextSize(4), 0, 0], ...
    'Tooltip', AutoscaleTooltip, ...
    'Callback', @refresh);

% Add some controls to the transform parameters panel.
EditSize = [0, 0, 0.5, 0.12];
TextSize = [0, 0, 0.5, 0.12];
PopupSize = [0, 0, 0.5, 0.12];
BasisTooltipString = ...
    sprintf(['Basis of the transformation.  Coordinate transforms\n', ...
    'compute the transform from localizations made from your\n', ...
    'fiducial images.  Image transforms compute a transform\n', ...
    'directly from your fiducial images (easier, but not as good)']);
uicontrol(TransformParametersPanel, 'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Transformation basis: ', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', BasisTooltipString, ...
    'Units', 'normalized', ...
    'Position', TextSize+[0, 1-TextSize(4), 0, 0]);
ControlHandles.BasisPopup = uicontrol(TransformParametersPanel, ...
    'Style', 'popupmenu', 'String', {'coordinates', 'images'}, ...
    'Tooltip', BasisTooltipString, ...
    'Units', 'normalized', ...
    'Position', PopupSize+[TextSize(3), 1-PopupSize(4), 0, 0], ...
    'Callback', @refresh);
TransformOptionsTooltip = ...
    sprintf(['Type of transform to be used.  The options listed\n', ...
    'will depend on the transformation basis.  See\n', ...
    '''doc fitgeotrans'' for descriptions of the coordinate\n', ...
    'transforms and ''doc imregtform'' for descriptions of the\n', ...
    'image transforms.']);
uicontrol(TransformParametersPanel, 'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Transformation type: ', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', TransformOptionsTooltip, ...
    'Units', 'normalized', ...
    'Position', TextSize+[0, 1-2*TextSize(4), 0, 0]);
ControlHandles.TransformTypePopup = uicontrol(TransformParametersPanel, ...
    'Style', 'popupmenu', 'String', obj.CoordTransformOptions, ...
    'Value', ...
    find(strcmp(obj.TransformationType, obj.CoordTransformOptions)), ...
    'Tooltip', TransformOptionsTooltip, ...
    'Units', 'normalized', ...
    'Position', PopupSize+[TextSize(3), 1-2*PopupSize(4), 0, 0], ...
    'Callback', @selectTransform);
TransformInputTooltip = ...
    sprintf(['See ''doc fitgeotrans''\n', ...
    '(or obj.NNeighborPoints/obj.PolynomialDegree)']);
ControlHandles.TransformInputText = uicontrol(TransformParametersPanel, ...
    'Style', 'text', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'N neighbor points: ', ...
    'HorizontalAlignment', 'right', ...
    'Tooltip', TransformInputTooltip, ...
    'Units', 'normalized', ...
    'Position', TextSize+[0, 1-3*TextSize(4), 0, 0], ...
    'Visible', 'off');
ControlHandles.TransformInputEdit = uicontrol(TransformParametersPanel, ...
    'Style', 'edit', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, ...
    'String', num2str(obj.NNeighborPoints), ...
    'HorizontalAlignment', 'center', ...
    'Tooltip', TransformInputTooltip, ...
    'Units', 'normalized', ...
    'Position', EditSize+[EditSize(3), 1-3*PopupSize(4), 0, 0], ...
    'Visible', 'off', 'Callback', @refresh);

% Add controls to the transform control panel.
ButtonSize = [0, 0, 1, 0.12];
ControlHandles.ComputeTransformButton = uicontrol(...
    TransformControlsPanel, ...
    'Style', 'pushbutton', 'BackgroundColor', [0, 1, 0], ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Compute transform', ...
    'Units', 'normalized', ...
    'Position', ButtonSize+[0, 1-ButtonSize(4), 0, 0], ...
    'Tooltip', sprintf(...
    ['Compute a transform based on the defined parameters.\n', ...
    'The transform will be stored in obj.RegistrationTransform.']), ...
    'Callback', @computeTransform);
ControlHandles.VisualizeTransformButton = uicontrol(...
    TransformControlsPanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Visualize transform magnitude', ...
    'Units', 'normalized', ...
    'Position', ButtonSize+[0, 1-3*ButtonSize(4), 0, 0], ...
    'Tooltip', ...
    'Visualize the magnitude of the transform (opens a new figure).', ...
    'Callback', @visualizeTransformMagnitude);
ControlHandles.VisualizeResultsButton = uicontrol(...
    TransformControlsPanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Visualize registration performance', ...
    'Units', 'normalized', ...
    'Position', ButtonSize+[0, 1-4*ButtonSize(4), 0, 0], ...
    'Tooltip', ...
    sprintf(['Visualize the registration performance on the\n', ...
    'fiducial localizations (only relevant for coordinate type\n', ...
    'transforms).']), ...
    'Callback', @visualizeRegistrationPerformance);
ControlHandles.VisualizeErrorButton = uicontrol(...
    TransformControlsPanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Visualize registration error', ...
    'Units', 'normalized', ...
    'Position', ButtonSize+[0, 1-5*ButtonSize(4), 0, 0], ...
    'Tooltip', ...
    'Visualize the registration error (opens a new figure).', ...
    'Callback', @visualizeError);
ControlHandles.ExportTransformButton = uicontrol(...
    TransformControlsPanel, ...
    'Style', 'pushbutton', ...
    'FontUnits', 'normalized', 'FontSize', 0.5, 'FontWeight', 'bold', ...
    'String', 'Export transform', ...
    'Units', 'normalized', ...
    'Position', ButtonSize, ...
    'Tooltip', ...
    'Save the transform (and related information) in a .mat file.', ...
    'Callback', @exportTransformCallback);

% Ensure the GUI items reflect class properties and vice versa.
refresh()


%{
This section contains various nested and callback functions for use in this
method.
%}

    function guiToProperties(~, ~)
        % This function takes relevant inputs in the GUI and sets them as
        % class properties.
        
        % Update obj.SplitFormat based on the popup menu selection (if
        % needed).
        if (numel(obj.SMF.Data.FileName) == 1)
            obj.SplitFormat = obj.SplitFormatOptions{...
                ControlHandles.ROISplittingPopup.Value};
        end
        
        % Update obj.ManualCull and obj.AutoscaleFiducials based on the
        % appropriate checkboxes.
        obj.ManualCull = ControlHandles.CullCheckbox.Value;
        obj.AutoscaleFiducials = ...
            ControlHandles.AutoscaleFiducialCheckbox.Value;
        
        % Set the transformation basis and type.
        obj.TransformationBasis = ControlHandles.BasisPopup.String{...
            ControlHandles.BasisPopup.Value};
        obj.TransformationType = ...
            ControlHandles.TransformTypePopup.String{...
            ControlHandles.TransformTypePopup.Value};
        
        % Set the additional transform inputs if needed (e.g.,
        % obj.NNeighborPoints).
        if strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'lwm')
            obj.NNeighborPoints = str2double(...
                ControlHandles.TransformInputEdit.String);
        elseif strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'polynomial')
            obj.PolynomialDegree = str2double(...
                ControlHandles.TransformInputEdit.String);
        end
    end

    function propertiesToGUI(~, ~)
        % This function updates the GUI based on the current class
        % properties.
        
        % Populate the fiducial file list if it is empty (re-populating
        % everytime we call this method is annoying, as it will change the
        % order of the files each time a new reference is selected).
        if isempty(ControlHandles.FileList.String)
            ControlHandles.FileList.String = obj.SMF.Data.FileName;
        end
        
        % Update the reference file/ROI selection GUI elements.
        if (numel(obj.SMF.Data.FileName) > 1)
            % Display the reference fiducial elements.
            ControlHandles.ReferenceFileText.Visible = 'on';
            ControlHandles.ReferenceFileName.Visible = 'on';
            ControlHandles.ROISplittingText.Visible = 'off';
            ControlHandles.ROISplittingPopup.Visible = 'off';
        else
            % Display the ROI selection elements.
            ControlHandles.ROISplittingText.Visible = 'on';
            ControlHandles.ROISplittingPopup.Visible = 'on';
            ControlHandles.ReferenceFileText.Visible = 'off';
            ControlHandles.ReferenceFileName.Visible = 'off';
        end
        if ~(isempty(obj.SMF.Data.FileName) ...
                || isempty(obj.SMF.Data.FileName{1}))
            ControlHandles.ReferenceFileName.String = ...
                obj.SMF.Data.FileName{1};
        else
            ControlHandles.ReferenceFileName.String = '';
        end
        
        % Update the checkbox for the manual cull and autoscale fiducial
        % options.
        ControlHandles.CullCheckbox.Value = obj.ManualCull;
        ControlHandles.AutoscaleFiducialCheckbox.Value = ...
                obj.AutoscaleFiducials;
        
        % Update the transformation basis and type.
        % NOTE: I have to reset the 'Value' of the popup to 1 (even if it
        %       changes a few lines later) to prevent conflicts (i.e.,
        %       Value>numel(String))
        ControlHandles.BasisPopup.String = obj.TransformationBasisOptions;
        ControlHandles.BasisPopup.Value = ...
            find(strcmp(obj.TransformationBasis, ...
            ControlHandles.BasisPopup.String));
        ControlHandles.TransformTypePopup.Value = 1;
        if strcmp(obj.TransformationBasis, 'coordinates')
            ControlHandles.TransformTypePopup.String = ...
                obj.CoordTransformOptions;
        else
            ControlHandles.TransformTypePopup.String = ...
                obj.ImageTransformOptions;
        end
        TransformIndex = find(strcmp(obj.TransformationType, ...
            ControlHandles.TransformTypePopup.String));
        if isempty(TransformIndex)
            TransformIndex = 1;
        end
        ControlHandles.TransformTypePopup.Value = TransformIndex;
        
        % Modify the text and edit box for the additional transform inputs
        % (e.g., NNeighborPoints).
        if strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'lwm')
            ControlHandles.TransformInputText.Visible = 'on';
            ControlHandles.TransformInputText.String = ...
                'N neighbor points: ';
            ControlHandles.TransformInputEdit.Visible = 'on';
            ControlHandles.TransformInputEdit.String = ...
                num2str(obj.NNeighborPoints);
        elseif strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'polynomial')
            ControlHandles.TransformInputText.Visible = 'on';
            ControlHandles.TransformInputText.String = ...
                'Polynomial degree: ';
            ControlHandles.TransformInputEdit.Visible = 'on';
            ControlHandles.TransformInputEdit.String = ...
                num2str(obj.PolynomialDegree);
        else
            ControlHandles.TransformInputText.Visible = 'off';
            ControlHandles.TransformInputEdit.Visible = 'off';
        end
    end

    function refresh(~, ~)
        % This function calls guiToProperties() and propertiesToGUI()
        guiToProperties()
        propertiesToGUI()
    end

    function addFile(~, ~)
        % This function allows the user to interactively select the
        % fiducial file(s).
        
        % Ensure obj is updated.
        guiToProperties()
        
        % Request the user to select the fiducial(s).
        [FileName, FileDir] = uigetfile({'*.mat'; '*.h5'}, ...
            'Select fiducial file(s)', 'Multiselect', 'on');
        
        % Store the filename(s) and filepath in obj.SMF.
        if isequal(FileName, 0)
            return
        else
            obj.SMF.Data.FileDir = FileDir;
            if ~iscell(FileName)
                FileName = {FileName};
            end
            if (isempty(obj.SMF.Data.FileName) ...
                    || isempty(obj.SMF.Data.FileName{1}))
                obj.SMF.Data.FileName = FileName;
            else
                obj.SMF.Data.FileName = [obj.SMF.Data.FileName; FileName];
            end
        end
        
        % Update the file list.
        ControlHandles.FileList.String = obj.SMF.Data.FileName;
        
        % Update the GUI.
        propertiesToGUI()
    end

    function removeFile(~, ~)
        % This function removes the file selected in the filename list.
        
        % Ensure obj is updated.
        guiToProperties()
        
        % Remove the selected file from the file list.
        FileIndex = ControlHandles.FileList.Value;
        IndexArray = 1:numel(obj.SMF.Data.FileName);
        KeepBool = (IndexArray ~= FileIndex);
        obj.SMF.Data.FileName = obj.SMF.Data.FileName(KeepBool);
        ControlHandles.FileList.Value = 1;
        ControlHandles.FileList.String = ...
            ControlHandles.FileList.String(KeepBool);
        
        % Update the GUI.
        propertiesToGUI()
    end

    function selectReference(Source, ~)
        % This function re-orders the files so that the selected file will
        % be used as the reference (internally, ChannelRegistration always
        % uses the first file as the reference).
        
        % Ensure obj is updated.
        guiToProperties()
        
        % Reorder the files so that the currently selected file is the
        % first element of obj.SMF.Data.FileName).
        FileIndex = Source.Value;
        IndexArray = 1:numel(Source.String);
        obj.SMF.Data.FileName = Source.String([FileIndex, ...
            setdiff(IndexArray, FileIndex)]);
        if ~isempty(obj.SMF.Data.FileName)
            ControlHandles.ReferenceFileName.String = ...
                obj.SMF.Data.FileName{1};
        else
            ControlHandles.ReferenceFileName.String = '';
        end
        
        % Update the GUI.
        propertiesToGUI()
    end

    function openSMFGUI(~, ~)
        % This function opens the SMF class GUI.
        guiToProperties()
        obj.SMF.gui();
        propertiesToGUI()
    end

    function selectTransform(~, ~)
        % This function modifies some GUI properties when the transform
        % type is selected.
        % NOTE: Even though refresh() would be ideal, I can't use that
        %       because it calls guiToProperties() first (the block below
        %       is found in propertiesToGUI()) which can cause an error
        %       when switching between 'lwm' and 'polynomial'.  We also
        %       don't want to just flip the order here, as that could cause
        %       other bugs.
        
        % Update the GUI.
        if strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'lwm')
            ControlHandles.TransformInputText.Visible = 'on';
            ControlHandles.TransformInputText.String = ...
                'N neighbor points: ';
            ControlHandles.TransformInputEdit.Visible = 'on';
            ControlHandles.TransformInputEdit.String = ...
                num2str(obj.NNeighborPoints);
        elseif strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'polynomial')
            ControlHandles.TransformInputText.Visible = 'on';
            ControlHandles.TransformInputText.String = ...
                'Polynomial degree: ';
            ControlHandles.TransformInputEdit.Visible = 'on';
            ControlHandles.TransformInputEdit.String = ...
                num2str(obj.PolynomialDegree);
        else
            ControlHandles.TransformInputText.Visible = 'off';
            ControlHandles.TransformInputEdit.Visible = 'off';
        end
        
        % Update the properties.
        if strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'lwm')
            obj.NNeighborPoints = str2double(...
                ControlHandles.TransformInputEdit.String);
        elseif strcmp(ControlHandles.TransformTypePopup.String{...
                ControlHandles.TransformTypePopup.Value}, 'polynomial')
            obj.PolynomialDegree = str2double(...
                ControlHandles.TransformInputEdit.String);
        end
        
        % Refresh both the GUI and obj.
        refresh()
    end

    function computeTransform(Source, ~)
        % This function will compute a registration transform.
        
        % Ensure obj is updated.
        guiToProperties()
        
        % Modify the compute transform button and then proceed to find the
        % transform.
        OriginalString = Source.String;
        Source.BackgroundColor = [1, 0, 0];
        Source.Enable = 'off';
        Source.String = 'Computing transform...';
        try 
            obj.findTransform();
        catch Exception
            Source.BackgroundColor = [0, 1, 0];
            Source.Enable = 'on';
            Source.String = OriginalString;
            rethrow(Exception)
        end
        disp(['Transform(s) computed and stored in ', ...
            'obj.RegistrationTransform'])
        Source.BackgroundColor = [0, 1, 0];
        Source.Enable = 'on';
        Source.String = OriginalString;
        
        % Update the GUI.
        propertiesToGUI()
    end

    function visualizeTransformMagnitude(~, ~)
        % This function creates a visualization of the computed 
        % transform(s).
        
        % Make sure a transform has already been computed.
        if (isempty(obj.RegistrationTransform) ...
                || isempty(obj.RegistrationTransform{2}))
            error(['You have to compute the transform using the ', ...
                'compute transform button before visualizing the ', ...
                'results (i.e., obj.RegistrationTransform must be ', ...
                'populated first).'])
        end
        
        % Determine how to proceed based on the transformation basis.
        % NOTE: If the user computes a transform and then changes the
        %       basis, this might not work correctly.
        for ii = 2:numel(obj.RegistrationTransform)
            % Note that obj.RegistrationTransform{1} isn't useful, it's
            % just there as sort of a placeholder.
            FigureName = sprintf('Transform %i', ii);
            if strcmp(obj.TransformationBasis, 'coordinates')
                obj.visualizeCoordTransform(figure('Name', FigureName), ...
                    obj.RegistrationTransform{ii}, ...
                    size(obj.FiducialImages, [1, 2]));
            else
                TempFigureHandle = figure('Name', FigureName);
                obj.visualizeImageTransform(...
                    axes(TempFigureHandle), ...
                    obj.RegistrationTransform{ii}, ...
                    size(obj.FiducialImages, [1, 2]));
            end
        end
        
    end

    function visualizeRegistrationPerformance(~, ~)
        % This function creates a visualization of the computed 
        % transform(s) on the fiducial localizations.
        
        % Make sure a transform has already been computed.
        if (isempty(obj.RegistrationTransform) ...
                || isempty(obj.RegistrationTransform{2}))
            error(['You have to compute the transform using the ', ...
                'compute transform button before visualizing the ', ...
                'results (i.e., obj.RegistrationTransform must be ', ...
                'populated first).'])
        end
        
        % Determine how to proceed based on the transformation basis.
        % NOTE: If the user computes a transform and then changes the
        %       basis, this might not work correctly.
        for ii = 2:numel(obj.RegistrationTransform)
            % Note that obj.RegistrationTransform{1} isn't useful, it's
            % just there as sort of a placeholder.
            if strcmp(obj.TransformationBasis, 'coordinates')
                FigureName = sprintf('Transform %i', ii);
                TempFigureHandle = figure('Name', FigureName);
                obj.visualizeRegistrationResults(...
                    axes(TempFigureHandle), ...
                    obj.RegistrationTransform{ii}, ...
                    obj.Coordinates{ii}(:, :, 2), ...
                    obj.Coordinates{ii}(:, :, 1), ...
                    obj.FiducialImages(:, :, ii), ...
                    obj.FiducialImages(:, :, 1));
            else
                warning(['You can only visualize the registration', ...
                    'performance for coordinate transforms.'])
                return
            end
        end
        
    end

    function visualizeError(~, ~)
        % This function calls obj.visualizeRegistrationError().
        
        % Make sure a transform has already been computed.
        if (isempty(obj.RegistrationTransform) ...
                || isempty(obj.RegistrationTransform{2}))
            error(['You have to compute the transform using the ', ...
                'compute transform button before visualizing the ', ...
                'results (i.e., obj.RegistrationTransform must be ', ...
                'populated first).'])
        end
        
        % Determine how to proceed based on the transformation basis.
        % NOTE: If the user computes a transform and then changes the
        %       basis, this might not work correctly.
        for ii = 2:numel(obj.RegistrationTransform)
            % Note that obj.RegistrationTransform{1} isn't useful, it's
            % just there as sort of a placeholder.
            if strcmp(obj.TransformationBasis, 'coordinates')
                FigureName = sprintf('Transform %i', ii);
                TempFigureHandle = figure('Name', FigureName);
                obj.visualizeRegistrationError(axes(TempFigureHandle), ...
                    obj.RegistrationTransform{ii}, ...
                    obj.Coordinates{ii}(:, :, 2), ...
                    obj.Coordinates{ii}(:, :, 1));
            else
                warning(['You can only visualize the registration', ...
                    'error for coordinate transforms.'])
                return
            end
        end
    end

    function exportTransformCallback(~, ~)
        % This callback calls obj.exportTransform().
        
        % Allow the user to select the export location and then export the
        % transform(s).
        FileDir = uigetdir('Specify save location for the transform(s)');
        if isequal(FileDir, 0)
            return
        else
            obj.exportTransform(FileDir);
        end
        
    end

end