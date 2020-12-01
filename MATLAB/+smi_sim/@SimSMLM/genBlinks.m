function [SMD_Model] = genBlinks(obj,SMD_True,K_OnToOff,K_OffToOn,K_OnToBleach,NFrames,StartState)

%This function generates the blinking time trace for a single particle 
%over the given number of the frames considering the input parameters and
%returns the SMD_Model. 

 % INPUTS:
 
 % [SMD_True]: This is a structure with the following three fields:
 
 % SMD_True.X: The X-positions of particles generated randomly.
 % (Number of the generated particles x 1)(Pixels)
    
 % SMD_True.Y: The Y-positions of particles generated randomly
 % (Number of the generated particles x 1),(Pixels)
    
 % SMD_True.Z: The Z-positions of particles generated randomly
 % (Number of the generated particles x 1), (um)
 
 % K_OnToOff: Fluorophore turns Off from On state (1 frames^-1)
 
 % K_OffToOn: Fluorophore return to On state from Off state (0.0005 frames^-1)
 
 % K_OnToBleach: Fluorophore bleached out (1/5 frames^-1)
 
 % NFrames: Number of frames (pixels x pixels)
 
 % StartState: A string which determine if the particle starts on or
 % starts randomly on or off. It can be either 'on' or 'Equib'.
 
 % OUTPUTS:
 
 % [SMD_Model]: This is a structure with the following fields:
 
 % SMD_Model.X: The X-positions of the particles seen on the frames.
 % (Number of the seen particles x 1),(Pixels)
    
 % SMD_Model.Y: The Y-positions of the particles seen on the frames.
 % (Number of the seen particles x 1),(Pixels)
    
 % SMD_Model.Z: The Z-positions of particles (um)
    
 % SMD_Model.Photons: The intensity of the particles.
 % (Number of the seen particles x 1),(Photon counts)
    
 % SMD_Model.FrameNum:The frames that the particles have been detected.
 % (Number of the seen particles x 1)
 
 % SMD_Model.PSFSigma: Point Spread Function Sigma size (Pixels)
 
 % SMD_Model.Bg: Background Count Rate (counts/pixel), this will be empty.

% First making empty vectors that will be filled later.
Photons=[];
X=[];
Y=[];
Z=[];
FrameNum=[];
PSFSigma=[];
Bg=[];
IntArray = zeros(obj.NLabels, NFrames);

% The following loop iterates over each particle to generate the blinking
% events for them.

for mm=1:obj.NLabels
    %genBlinks() makes the blinking events.
    Temp=Blinks(K_OnToOff,K_OffToOn,K_OnToBleach,NFrames,StartState);
    IntArray(mm,:)=Temp';
    %Finding the frames where the particle was ON. Note that the
    %particle might not be on at all and this would be an empty
    %array. In this case, we won't have any FrameNum, Photons or
    %found positions for this particle.
    FrameNumIndiv = find(Temp~=0);
    if ~isempty(FrameNumIndiv) 
        FrameNum = cat(1,FrameNum,FrameNumIndiv);
        Indiv = obj.EmissionRate*Temp(FrameNumIndiv);
        Photons = cat(1,Photons,Indiv);
        Indiv(:,1)=obj.LabelCoords(mm,1);
        X = cat(1,X,Indiv);
        Indiv(:,1)=obj.LabelCoords(mm,2);
        Y = cat(1,Y,Indiv);
    end
    SMD_Model.X = X;
    SMD_Model.Y = Y;
    if isscalar (obj.PSFSigma) 
        SMD_Model.Z = [];
        SMD_Model.PSFSigma = obj.PSFSigma*ones([length(Photons),1]);
    end
    SMD_Model.FrameNum = FrameNum;
    SMD_Model.Photons = Photons;
    SMD_Model.Bg = 0;
end
    
    %Nested function to generate blinking events.
    function IvsT=Blinks(K_OnToOff,K_OffToOn,K_OnToBleach,NFrames,StartState)
    %genBlinks() generates blinking time trace for a single
    %particle over the given number of the frames considering
    %the parameters K_OffToOn, K_OnToOff and K_OnToBleach.
    
    NPairs=0; %Number of the times that the particle goes on.
    
    %Based on the input 'StartState' the particle can start in the
    %on-state or it can be started randomly in either on-state or
    %off-state.
    switch StartState
        case 'On'
            T=0; %The time when the particle goes on. T=0 implies
            %that the particle is on from the beginning.
        case 'Equib'
            %Find start state:
            State=rand < (K_OffToOn/(K_OffToOn+K_OnToOff));
            %Randomly determine if the particle starts off or on.
            if State %starts on
                T=0;
            else %starts from off
                T=exprnd(1/K_OffToOn); %The random time when the particle goes on.
            end
    end
    
    %The following while-loop gives the times when the particle
    %goes on, goes off and photobleach.
    
    while T<NFrames
        NPairs=NPairs+1;
        % OnOffPairs is an array of 3 columns, where the first column gives
        % the times when the particle goes on, the second column gives the
        % time when the particle goes off and third column gives the time
        % when particle photobleaches.
        OnOffPairs(NPairs,1)=T;
        D=exprnd(1/K_OnToOff); %Generate blink duratrion
        OnOffPairs(NPairs,2)=min(T+D,NFrames); %The On-time plus the duration gives the off-time.
        OnOffPairs(NPairs,3)= rand > (K_OnToOff/(K_OnToOff+K_OnToBleach)); %fluorophore bleaches
        %if this is condition met.
        T=T+D+exprnd(1/K_OffToOn); %Advance to the next blinking event.
    end
    
    %Turn blinking events to I vs T
    IvsT=zeros(NFrames,1);
    for nn=1:NPairs
        StartT=floor(OnOffPairs(nn,1))+1; %index for start frame
        EndT=min(NFrames,floor(OnOffPairs(nn,2))+1); %index for start frame
        if StartT==EndT %Blinking happens within one frame
            IvsT(StartT)=OnOffPairs(nn,2)-OnOffPairs(nn,1);
        else
            %This for-loop goes over the frames where the particle is on.
            for ii=StartT:EndT
                if ii==StartT
                    IvsT(ii)=StartT-OnOffPairs(nn,1);
                elseif ii==EndT
                    IvsT(ii)=1-(EndT-OnOffPairs(nn,2));
                else
                    IvsT(ii)=1;
                end
            end
        end
    end

    end % Blinks
end % genBlinks
