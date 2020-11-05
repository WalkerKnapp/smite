function unitTest()

   clear
   obj=smi_sim.SimSMLM();
   obj.SZ = 256;
   obj.Rho=10;
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
   [Model] = smi_sim.SimSMLM.gaussBlobImage(SZ,NFrames,SMD_Model,Bg,0,0)
   dipshow(SMD_Model)
end


 % NoiseIm = Bg*ones(SZ); Noise factor will be included later in Data. 
