function unitTest()

   clear
   obj=smi_sim.SimSMLM();
   obj.SZ = 256;
   obj.Rho=100;
   obj.NFrames=10;
   obj.ZoomFactor=1;
   obj.K_OnToOff=1;
   obj.K_OffToOn=0.005;
   obj.K_OnToBleach=0.2;
   obj.EmissionRate=1000;
   obj.Bg=15;
   obj.PSFSigma=1.3;
   [SMD_True] = obj.simStar(16);
   [SMD_Model] = obj.genBlinks(SMD_True,1,0.005,0.2,10,'Equib'); 
   
   % To generate the blobs, execute the following:
   % smi_sim.GaussBlobs.gaussBlobImage(SZ,NFrames,SMD,Background,Density,VarianceIm)
   % In order to generate blobs, I set following parameters:
   [Model] = smi_sim.GaussBlobs.gaussBlobImage(obj.SZ,obj.NFrames,SMD_Model,0,0,0)
   dipshow(Model)
end


 
