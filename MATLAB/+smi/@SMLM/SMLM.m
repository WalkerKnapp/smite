classdef SMLM < handle
% Single Molecule Localization Microscopy Analysis
%
% This is a high-level class that provides complete analysis of SMLM data.
% fullAnalysis/testFit performs an analysis on all/a selected dataset(s).
% These routines require a Single Molecule Fitting (SMF) structure describing
% the data and defining the analysis parameters.  This structure is either
% produced earlier and fed into SMLM or it is created interactively when SMLM
% is invoked.  The output is a Results.mat file holding the SMF and Single
% Molecule Data (SMD) structures, the latter containing the processed data.  In
% addition, various plots describing the processed data are created and placed
% in a directory under Results identifying the dataset.  This identification is
% derived from the original dataset's name, optionally with an analysis ID
% appended.  See generatePlots for more details on the plots produced.

% =========================================================================
properties
    SMDPreThresh      % Keeps track of why localizations were filtered out
    SMD               % SMD structure with final analysis results
    SMF               % Single Molecule Fitting structure
    FitFramePreFC     % Fits per frame pre-frame connection
    PlotDo = [] % Plots to generate (all by default);see generatePlots comments
    SRImageZoom  = 20 % magnification factor for SR     images generated
    SRCircImZoom = 25 % magnification factor for circle images generated
    Verbose = 1       % Verbosity level
    VerboseTest = 3   % Verbosity level for testFits
    FullvsTest        % Logical value set by fullAnalysis or testFit to tell
                      % saveResults to make the proper call to generatePlots
    CalledByGUI=false % Keeps track of how fitting is called
end
% =========================================================================

% =========================================================================
properties (Hidden)
    DC                % DriftCorrection class object used internally

    % Top level results directory: A few special results/plots (like GaussIm)
    % are saved here.  Default value is obj.SMF.Data.ResultsDir set in testFit
    % and fullAnalysis.  The rest of the results/plots are saved in
    % ResultsSubDir which will be a subdirectory of ResultsDir; its name will
    % be derived from the dataset name and analysis ID.
    ResultsDir = []   % This is set from SMF.Data.ResultsDir
end % properties (Hidden)
% =========================================================================

% =========================================================================
methods

    function obj=SMLM(SMF,StartGUI)
        %SMLM Create an SMLM object
        %
        % INPUTS:
        %   SMF: Single Molecule Fitting Structure (Optional)
        %   StartGUI:   Automatically open the GUI (0 or 1)
        %
        % OUTPUT:
        %   SMLMobj: SMLM object. Autogenerated if nargout==0
        %
        %   If no inputs are given, a GUI is opened by default

        switch nargin
            case 0
                obj.SMF=smi_core.SingleMoleculeFitting();
                StartGUI=1;
            case 1
                obj.SMF = SMF;
                StartGUI=0;
            case 2
                obj.SMF = SMF;
        end

        %Autonaming
        if nargout==0 % No name has been given
            varname = 'SMLMobj'; % try using SR
            if exist(varname,'var') %if that fails try SR1,SR2, ...
                n=1;
                s=sprintf('%s%d',varname,n);
                while exist(s,'var')
                    s=sprintf('%s%d',varname,n);
                    n=n+1;
                end
                assignin('base',s,obj);
            else
                assignin('base',varname,obj);
            end
        end

        if StartGUI
            obj.gui();
        end

    end

    % ---------------------------------------------------------------------

    function fullAnalysis(obj)
        % fullAnalysis analyzes all data, saves results and plots.

        obj.ResultsDir = obj.SMF.Data.ResultsDir;

        obj.FullvsTest = true;
        if isempty(obj.SMF.Data.DatasetList)
           obj.analyzeAll();
        else
           obj.analyzeAll(obj.SMF.Data.DatasetList);
        end
        obj.saveResults();

        if obj.Verbose >= 1
            fprintf('Done fullAnalysis.\n');
        end

    end

    % ---------------------------------------------------------------------

    function testFit(obj, DatasetIndex)
        % testFit performs detailed analysis and feedback of one dataset.
        
        obj.ResultsDir = obj.SMF.Data.ResultsDir;

        obj.FullvsTest = false;
        obj.analyzeAll(DatasetIndex);
        obj.saveResults();
        
        if obj.Verbose >= 1
            fprintf('Done testFit.\n');
        end

    end

    % ---------------------------------------------------------------------

    function analyzeAll(obj, DatasetList)
        % analyzeAll loops over a list of datasets and creates an SMD.
        % If DatasetList not provided, use obj.SMD.Data.DatasetList .
        % analyzeAll flow:
        %
        % analyzeAll:
        %    for n = DatasetList
        %       analyzeDataset:
        %          LoadData           (load raw data)
        %          DataToPhotons      (gain and offset corrections)
        %          LocalizeData       (produce localizations for SMD structure)
        %             Threshold       (thresholding of localizations generated)
        %          SingleMoleculeData (catSMD for SMDPreThresh)
        %          FrameConnection    (frame connection)
        %          DriftCorrection    (intra-dataset)
        %       SingleMoleculeData    (catSMD for SMD)
        %    end
        %    DriftCorrection (inter-dataset)
        %    Threshold       (rejected localization statistics)

        % Define the list of datasets to be processed.
        obj.SMF = smi_core.LoadData.setSMFDatasetList(obj.SMF);
        % DatasetList takes priority over what is in SMF.
        if ~exist('DatasetList', 'var')
            DatasetList = obj.SMF.Data.DatasetList;
        else
            obj.SMF.Data.DatasetList = DatasetList;
        end

        % Initialize FitFramePreFC for this invocation of analyzeAll.
        obj.FitFramePreFC = cell(max(DatasetList), 1);

        % DriftCorrection class object is also used in analyzeDataset.
        obj.DC = smi_core.DriftCorrection(obj.SMF);
        obj.DC.Verbose = obj.Verbose;
        obj.SMD=[];
        obj.SMDPreThresh=[];
        if obj.Verbose >= 1
            fprintf('Processing %d datasets ...\n', numel(DatasetList));
        end
        for nn=1:numel(DatasetList)
            SMDnn = obj.analyzeDataset(DatasetList(nn), nn);
            obj.SMD=smi_core.SingleMoleculeData.catSMD(obj.SMD,SMDnn,false);
        end

        % Inter-dataset drift correction.
        if obj.SMF.DriftCorrection.On && numel(DatasetList) > 1
            if obj.Verbose >= 1
                fprintf('Drift correcting (inter-dataset) ...\n');
            end
            obj.SMD = obj.DC.driftCorrectKNNInter(obj.SMD);
        end

        % Copy PixelSize from SMF to SMD.
        obj.SMD.PixelSize = obj.SMF.Data.PixelSize;

        % Produce some statistics on rejected localizations.
        THR = smi_core.Threshold;
        if obj.Verbose >= 1 && obj.SMF.Thresholding.On
           THR.rejectedLocalizations(obj.SMDPreThresh, '');
        end
    end

    % ---------------------------------------------------------------------

    function SMD=analyzeDataset(obj,DatasetIndex,DatasetCount)
        % analyzeDataset loads and analyzes one dataset.

        if ~exist('DatasetCount', 'var')
            DatasetCount = 1;
        end

        if obj.Verbose >= 1
            fprintf('Loading dataset %d ...\n', DatasetIndex);
        end
        LD = smi_core.LoadData;
        [~, Dataset, obj.SMF]=LD.loadRawData(obj.SMF,DatasetIndex);

        % Perform the gain and offset correction.
        DTP = smi_core.DataToPhotons(obj.SMF, Dataset, [], [], obj.Verbose);
        ScaledDataset = DTP.convertData();
        
        % Generate localizations from the current Dataset.
        if obj.FullvsTest
           V = obj.Verbose;
        else
           V = obj.VerboseTest;
        end
        LD = smi_core.LocalizeData(ScaledDataset, obj.SMF, V);
        if obj.Verbose >= 1
            fprintf('Generating localizations ...\n');
        end
        [SMD] = LD.genLocalizations();

        % Keep track of why localizations were filtered out.
        obj.SMDPreThresh = smi_core.SingleMoleculeData.catSMD( ...
                              obj.SMDPreThresh, LD.SMDPreThresh, false);

        % Define NDatasets, and DatasetNum from the dataset count.
        SMD.NDatasets  = 1;
        SMD.DatasetNum = DatasetCount * ones(size(SMD.FrameNum));

        % Perform frame-connection on localizations in SMD.
        obj.FitFramePreFC{DatasetIndex} = obj.fitsPerFrame(SMD, DatasetIndex);
        if obj.SMF.FrameConnection.On
            FC = smi_core.FrameConnection(SMD, obj.SMF, obj.Verbose);
            SMD = FC.performFrameConnection();
        end

        % Intra-dataset drift correction.
        if obj.SMF.DriftCorrection.On
            if obj.Verbose >= 1
                fprintf('Drift correcting (intra-dataset) ...\n');
            end
            SMD = obj.DC.driftCorrectKNNIntra(SMD, DatasetCount, DatasetIndex);
        end
    end

    % ---------------------------------------------------------------------

    function saveResults(obj)
        % saveResults saves all results and plots in subfolder.
        % gaussblobs, drift image, fits/frame, NumConnected hist,
        % Driftcorrection plots, precision hist, intensity hist,
        % mat file with SMD and SMF structures.
        % saveResults flow:
        %
        % saveResults:
        %    Save SMD and SMF structures
        %    generatePlots (plots saved for fullAnalysis, displayed for testFit)

        if isempty(obj.SMD)
            error('No SMD results structure found to save!');
        end

        [~, f, ~] = fileparts(obj.SMF.Data.FileName{1});
        if isempty(obj.SMF.Data.AnalysisID)
            fn = [f, '_Results.mat'];
            SubDir = f;
        else
            fnextend = strcat('_', obj.SMF.Data.AnalysisID, '_Results.mat');
            fn = [f, fnextend];
            SubDir = [f, '_', obj.SMF.Data.AnalysisID];
        end

        if obj.FullvsTest   % fullFit
           ResultsDir = obj.SMF.Data.ResultsDir;
           ResultsSubDir = fullfile(ResultsDir, SubDir);
           ShowPlots = false;
        else   % testFit
           ResultsDir = [];
           if obj.CalledByGUI || obj.VerboseTest >= 5
              ResultsSubDir = [];
              ShowPlots = true;
              % Reset CalledByGUI for the next call that comes here.
              obj.CalledByGUI = false;
           else
              ResultsSubDir = fullfile(obj.SMF.Data.ResultsDir, ...
                                       SubDir, 'TestFit');
              ShowPlots = false;
           end
        end

        SMD = obj.SMD;
        SMF = obj.SMF;
        if obj.Verbose >= 1
            fprintf('Saving SMD and SMF structures ...\n');
        end
        if ~isempty(ResultsDir) && ~isfolder(ResultsDir)
            mkdir(ResultsDir);
        end
        if ~isempty(ResultsSubDir) && ~isfolder(ResultsSubDir)
            mkdir(ResultsSubDir);
        end

        % Save SMD and SMF structures.
        if ~isempty(ResultsSubDir)
           save(fullfile(ResultsSubDir, fn), 'SMD', 'SMF', '-v7.3');
        end
        % Generate (and optionally) save various data plots/histograms.
        obj.generatePlots(ResultsDir, ResultsSubDir, obj.SMF.Data.AnalysisID,...
                          ShowPlots, obj.PlotDo);
    end

    % ---------------------------------------------------------------------

    generatePlots(obj, PlotSaveDir1, PlotSaveDir2, AnalysisID, ...
                       ShowPlots, PlotDo)
    gui(obj)
end % methods
% =========================================================================

% =========================================================================
methods(Static)
    FitFrame = fitsPerFrame(SMD, DatasetIndex)
    Success = unitTest()
end % methods(Static)
% =========================================================================

end
