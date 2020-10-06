classdef PointSpreadFunction < handle
%PointSpreadFunction Create and Quantify Point Spread Functions
%
%   Notes on Zernike Coeffiencts:  
%       We will use the Noll Linear Ordering Scheme.  Conversion from M,N
%       to Noll linear index can be done with: 
%       smi_psf.Zernike.zNoll2NM
%       smi_psf.Zernike.zNM2Noll
%       smi_psf.Zernike.zWyant2Noll
%
% REQUIRES:
%   Statistics Toolbox
%   Parallel Procesing Toolbox
%   NVidia GPU
%
% SEE ALSO:
%   smi_psf.Zernike
    
    properties
        Lambda              %Emission wavelength (micron)
        NA                  %Numerical Aperture
        N                   %Index of Refraction
        PSF                 %Point Spread Function
        PixelSize           %PixelSize of PSF (micron)
        Z                   %Array of Z positions (micron)
        DataSet             %PSF Data (SZxSZxNxM) N repeats of M focal planes
        DataFile            %Data file containing DataSet,Z. Optional: PixelSize,NA,N
        PSFStruct           %PSF Structure for PSF Model
        PSFModel            %PSF Model generated by PFStruct
        RawData             %Raw Data From File
        PSFData             %Cropped PSF Data (not averaged)
        SZ                  %Cropped PSF Size
        PSFCenter           %Center of PSF (Pixels) (Y,X)
        MaxZCMag=81         %Max Zenike Expansion in Magnitude Smoothing
        MaxZCPhase=81       %Max Zenike Expansion in Phase Smoothing
    end

    methods
        function obj=PointSpreadFunction()
            %Constructor
        end
        
        function loadData(obj,FileName,DataSetName)
            %loadData Load PSF Data and properties from *.mat file
            %
            %   File must contain PSF Data and Z, the vector of defocus
            %   values. 
            %
            %   PixelSize, NA, N will be loaded if avaiable in file. 
            %
            % INPUTS:
            %   FileName:       Name of *.mat file containing PSF Data (optional)
            %   DataSetName:    Name of PSF Data Variable Name (Default='DataSet') 
            %   
            if nargin>1
                obj.DataFile=FileName;
            else
                [FileName,PathName]=uigetfile('Y:\','*.mat');
                obj.DataFile=fullfile(PathName,FileName);
            end
            
            M=matfile(obj.DataFile);
            obj.DataSet=M.DataSet;
            obj.Z=M.Z;
            try
                obj.NA=M.NA;
            end
            try
                obj.N=M.N;
            end
            try
                obj.PixelSize=M.PixelSize;
            end
        end
        
        function savePSF(obj,FileName)
        %savePSF saves PSFStruct and PSF in a user-specified file
            if nargin>1
                SaveFile=FileName;
            else
                [FileName,PathName]=uigetfile('Y:\','*.mat');
                SaveFile=fullfile(PathName,FileName);
            end
            PSF=obj.PSFModel;
            save(SaveFile,'PSFStruct','PSF')
        end
        
        function cropData(obj,Center)
            %cropData Crop data to obj.SZ around Center
            %
            % If Center is not given and obj.PSFCenter is empty,
            % Center finding is interactive
            %
            % INPUTS
            %   Center:     Center of PSF (Pixels) (Y,X) (Default=obj.PSFCenter) 
            
            Data=mean(obj.RawData,3);
            if ((nargin<2) && (isempty(obj.PSFCenter)))
                h=dipshow(Data);
                C=dipgetcoords(h);
                close(h);
                obj.PSFCenter=[C(2),C(1)]+1;
            else
                obj.PSFCenter=Center;
            end
            
            obj.PSFData=obj.RawData(CTR(1)-obj.SZ/2+1:CTR(1)+obj.SZ/2,CTR(2)-obj.SZ/2+1:CTR(2)+obj.SZ/2,:);
    
        end
        
        function phaseRetrieve(obj)
            %phaseRetrieve Retrieval of Pupil Magnitude and Phase
            %
            %   See PointSpreadFunction.phaseRetrieval
            %    
            %
            
            %Crop Data and Average
            if isempty(obj.PSFData)
                obj.cropData();
            end
            
            Data=mean(obj.PSFData,3);
            
            %Make PSF Struct
            P=PointSpreadFunction.createPSFStruct();
            P.Z=obj.Z;
            P.Lambda=obj.Lambda;
            P.NA=obj.NA;
            P.N=obj.N;
            P.PixelSize=obj.PixelSize;
            P.SZ=obj.SZ;
            
            [obj.PSFStruct,obj.PSFModel]=...
                phaseRetrieval(P,Data,obj.MaxZCMag,obj.MaxZCPhase);
        end
    end
    
    methods (Static)
        
        %Data Structures
        [PSFStruct]=createPSFStruct()
        [ZStruct]=createZernikeStruct(SZ,Radius,NMax)
        
        %PSF Generation
        [PSF,PSFStruct]=scalarPSF(PSFStruct)
        [PSF,PSFStruct]=scalarPSFPupil(PSFStruct)
        [PSF,PSFStruct]=scalarPSFZernike(PSFStruct)
        [PSF,PSFStruct]=scalarPSFPrasadZone(PSFStruct,L)
        [PSF,PSFStruct]=oversamplePSFPupil(PSFStruct,Sampling)
        
        %CRLB Calculations
        [CRLB,DET]=crlbPSFPupil(PSFStruct,Photons,Bg,PlotFlag)
       
        %PSF Modeling
        [Pupil,PSF]=phaseRetrieval(PSFStruct,Data,MaxZCMag,MaxZCPhase)
        [PSFStruct]=phaseRetrievalEM(PSFStruct,Data)
        [PSFStruct,PSF]=phaseRetrieval_Spiral(PSFStruct,Data,MaxZCMag,MaxZCPhase)
        %Optimization
        [PSFStruct,PSF,CRLB]=optimPSFZernike(PSFStruct,PhaseMask,StartPhase,Photons,Bg)
        [OTFVector]=rescaleOTF(PSFStruct,Data)
        
        %Zernike Calculations
        [Image]=zernikeImage(NollCoef,SZ,Radius,R,Theta,Mask)  
        [Image]=zernikeSum(NollCoefs,ZStruct)    
        [NollCoef,ImageZ]=zernikeExpansion(Image,ZStruct)
        
        [Model,Data]=psfROIStack(SZ,SMD,PSF,XYSamPerPix,ZSamPerUnit,NoiseIm)

        %Unit Tests
        [CRLB]=crlbPSFPupil_unitTest()
        [Report]=optimPSFZernike_unitTest()
        [Report]=oversamplePSFPupil_unitTest()
        [PSFStruct]=phaseRetrieval_unitTest()
        [Report]=scalarPSFPrasadZone_unitTest()
        psfROIStack_unitTest()
        [Report]=zernikeImage_unitTest()
        unitTest()   % overall unitTest

    end

end
