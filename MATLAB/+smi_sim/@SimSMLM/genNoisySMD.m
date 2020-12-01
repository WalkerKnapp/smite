function [SMD_Data] = genNoisySMD(obj,SMD_Model)

% This function takes the SMD_Model struct as an input, generates the noisy 
% SMD struct and return SMD_Data with the following fields: 

% OUTPUTS:

% SMD_Data.X
% SMD_Data.Y
% SMD_Data.PSFSigma
% SMD_Data.FrameNum
% SMD_Data.Photons
% SMD_Data.Bg

X_SE=(SMD_Model.PSFSigma)/sqrt(SMD_Model.Photons);
SMD_Data.X=SMD_Model.X+randn(size(X_SE))*X_SE;

Y_SE=(SMD_Model.PSFSigma)/sqrt(SMD_Model.Photons);
SMD_Data.Y=SMD_Model.Y+randn(size(Y_SE))*Y_SE;

SMD_Data.PSFSigma=SMD_Model.PSFSigma;

SMD_Data.Photons=SMD_Model.Photons;
SMD_Data.FrameNum=SMD_Model.FrameNum;

SMD_Data.Bg=obj.Bg*ones(obj.SZ);

end

