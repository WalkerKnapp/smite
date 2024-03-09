function [Chain]=BaGoL_RJMCMC(SMD,Xi,SigAlpha,PMove,NChain,NBurnin,DEBUG, ...
    Mu_X, Mu_Y, Alpha_X, Alpha_Y)
%BaGoL_RJMCMC BaGoL's core RJMCMC algorithm
% [Chain]=BaGoL.BaGoL_RJMCMC(SMD,Xi,MaxAlpha,PMove,NChain,NBurnin,DEBUG)
%
% This the core BaGoL algorithm. It uses Reversible Jump Markov Chain 
% Monte Carlo to add, remove and move emitters, and to explore the 
% classification of localizations to emitters.  
%
% Prior information on the distribution of localizations per emitter is 
% required. The prior distribution is parameterized by either a Poisson or 
% Gamma distribution function.  For the Poisson distribution, the parameter
% [lambda] is the mean number of localizations per emitter.  For the Gamma
% distribution, [k, theta] are the shape and scale parameters.  k*theta
% is the mean localizations per emitter. 
%
% A linear drift of individual emitters can be included by allowing
% MaxAlpha to be non-zero. Drift velocities are given a uniform prior for
% each dimension from -MaxAlpha to MaxAlpha. 
%
% The output chain is a structure array of the post burn-in states 
% for the input subregion. Each element of the array contains fields 
% associated to a single accepted proposed parameter value. A description 
% of the fields is given below. 
%
% INPUTS:
%    SMD:      SMD structure with the following fields:
%       X:     X localization coordinates. (nm) (Nx1)
%       Y:     Y localization coordinates. (nm) (Nx1)
%       X_SE:  X localization precisions.  (nm) (Nx1)
%       Y_SE:  Y localization precisions.  (nm) (Nx1)
%       FrameNum:   localization frame numbers. (Nx1)
%    Xi:       Loc./emitter params [lambda] (Poisson) or [k theta] (Gamma) 
%    SigAlpha: Sigma of drift velocity. (nm) (Default = 0)
%    PMove:    Probabilities of proposing different moves in RJMCMC:
%              [1] Move Mu, Alpha
%              [2] Reallocation of Z
%              [3] Add
%              [4] Remove
%              (1x4) (Default = [0.25, 0.25, 0.25, 0.25])
%    NChain:   Length of the chain after the burn in. (Default = 2000)
%    NBurnin:  Length of the chain for burn in. (Default = 3000)
%    DEBUG:    0 or 1. Show an animation of the chain. (Default = 0)
%
% OUTPUT:
%    Chain:    Structure array of post burn-in states of the RJMCMC Chain. 
%       N: Number of emitters (Scalar)
%       X: X coordinate of emitters (Kx1)
%       Y: Y coordinate of emitters (Kx1)
%       AlphaX: Corresponding X drift velocities (Kx1)
%       AlphaY: Corresponding Y drift velocities (Kx1)
%       ID: Allocation parameter representing assigning a localization to 
%           an emitter. The order is the same as SMD.X (see above) (Nx1) 
%
% CITATION: "Sub-Nanometer Precision using Bayesian Grouping of Localizations"
%           Mohamadreza Fazel, Michael J. Wester, Sebastian Restrepo Cruz,
%           Sebastian Strauss, Florian Schueder, Thomas Schlichthaerle, 
%           Jennifer M. Gillette, Diane S. Lidke, Bernd Rieger,
%           Ralf Jungmann, Keith A. Lidke
%

% Created by: 
%    Mohamadreza Fazel and Keith A. Lidke (Lidkelab 2020)

%DEBUG=0;

DX = 1;
X_min = min(SMD.X-3*SMD.X_SE);
X_max = max(SMD.X+3*SMD.X_SE);
X_range = X_min:DX:X_max;
Y_min = min(SMD.Y-3*SMD.Y_SE);
Y_max = max(SMD.Y+3*SMD.Y_SE);
Y_range = Y_min:DX:Y_max;

[Xg,Yg] = meshgrid(X_range,Y_range);
PDFgrid = zeros(size(Xg));
for pp = 1:length(SMD.X)
    PDFgrid = PDFgrid + normpdf(Xg,SMD.X(pp),SMD.X_SE(pp)).*normpdf(Yg,SMD.Y(pp),SMD.Y_SE(pp)); 
end

CDF = cumsum(PDFgrid(:)/sum(PDFgrid(:)));
PDFgrid = PDFgrid/sum(PDFgrid(:));
Area = sum(sum(PDFgrid>max(PDFgrid(:))/1000));

Chain(NChain).N = [];
Chain(NChain).X = [];
Chain(NChain).Y = [];
Chain(NChain).AlphaX = [];
Chain(NChain).AlphaY = [];
Chain(NChain).ID = [];

if nargin<3
    MaxAlpha=0;
end

if nargin<4
    PMove = [.25 .25 .25 .25]; %PMove = [Theta Z Birth Death]
end

if nargin<5
    NChain = 3e3; %Total
end

if nargin<6
    NBurnin = 2e3;
end

%Storage of Chain
N=length(SMD.X);


%Initial Locations
if nargin < 8
    %Intial K Guess
    K=ceil(N/prod(Xi));
    Mu_X =SMD.X(randi(N,[1 K]))';
    Mu_Y =SMD.Y(randi(N,[1 K]))';
else
    K = length(Mu_X);
end

%Initial Alphas
if nargin < 10
    Alpha_X = zeros([1 K]);
    Alpha_Y = zeros([1 K]);
end

if N < 100
   LengN = 100; 
else
   LengN = N; 
end
%Calculating the Prior
if length(Xi)>1
   Gamma_K=Xi(1);
   Gamma_Theta=Xi(2);
   Pk=gampdf(N,(1:LengN)*Gamma_K,Gamma_Theta);
else
   Pk=poisspdf(N,(1:LengN)*Xi); 
end
Pk = Pk/sum(Pk);

% Intial Allocation
Z=Gibbs_Z(SMD,K,Mu_X,Mu_Y,Alpha_X,Alpha_Y);


% Run Chain
for nn=1:NChain+NBurnin
    if mod(nn, 100) == 0
        fprintf("%d\n", nn);
    end
    %Get move type:
    JumpType=length(PMove)+1-sum(rand<cumsum(PMove));

    K = length(Mu_X);

    % Remove emitters with no localizations mapped to them
    % Don't allow removal if we are doing MCMC (no add/remove)
    if PMove(4) > 0
        emitters_to_remove = setdiff(1:K, Z);
        for jj = length(emitters_to_remove):-1:1
            ii = emitters_to_remove(jj);

            Mu_X(ii)=[];
            Mu_Y(ii)=[];
            Alpha_X(ii)=[];
            Alpha_Y(ii)=[];
            K = length(Mu_X);%K-1;
            Z(Z>ii) = Z(Z>ii) - 1;
        end 
    end

    switch JumpType
        case 1  %Move Mu, Alpha 
            Mu_XTest=Mu_X;
            Mu_YTest=Mu_Y;
            Alpha_XTest=Alpha_X;
            Alpha_YTest=Alpha_Y;

            %Get new Mu and Alpha using Gibbs
            if SigAlpha>0
                for ID=1:K
                    [Mu_XTest(ID),Alpha_XTest(ID)]=Gibbs_MuAlpha(ID,Z,SMD.X,SMD.FrameNum,SMD.X_SE,SigAlpha);
                    [Mu_YTest(ID),Alpha_YTest(ID)]=Gibbs_MuAlpha(ID,Z,SMD.Y,SMD.FrameNum,SMD.Y_SE,SigAlpha);
                end
            else
                [Mu_XTest, Mu_YTest] = Gibbs_Mu_Bulk(K, Z, SMD.X, SMD.Y, SMD.X_SE, SMD.Y_SE);
            end
            
            Mu_X = Mu_XTest';
            Mu_Y = Mu_YTest';
            Alpha_X = Alpha_XTest;
            Alpha_Y = Alpha_YTest;
            
            if nn>NBurnin %Then record in chain
                
                Chain(nn-NBurnin).N = K;
                Chain(nn-NBurnin).X = Mu_X';
                Chain(nn-NBurnin).Y = Mu_Y';
                Chain(nn-NBurnin).AlphaX = Alpha_X';
                Chain(nn-NBurnin).AlphaY = Alpha_Y';
                Chain(nn-NBurnin).ID = Z;
                
            end

        case 2  %Reallocation of Z
            
            [ZTest]=Gibbs_Z(SMD, K,Mu_X,Mu_Y,Alpha_X,Alpha_Y);
            %Always accepted
            Z = ZTest;
                        
            if nn>NBurnin %Then record in chain
                
                Chain(nn-NBurnin).N = K;
                Chain(nn-NBurnin).X = Mu_X';
                Chain(nn-NBurnin).Y = Mu_Y';
                Chain(nn-NBurnin).AlphaX = Alpha_X';
                Chain(nn-NBurnin).AlphaY = Alpha_Y';
                Chain(nn-NBurnin).ID = Z;
                
            end
            
        case 3  %Add
                        
%           Sample the Emitter location from SR data
            ID = find(CDF>rand(),1);
            if isempty(ID)
                ID = length(CDF); 
            end
            [Ydraw,Xdraw]=ind2sub(size(PDFgrid),ID);
            Mu_XTest = cat(2,Mu_X,Xdraw+X_min-1);
            Mu_YTest = cat(2,Mu_Y,Ydraw+Y_min-1);      

            if SigAlpha>0
                Alpha_XTest = cat(2,Alpha_X,SigAlpha*randn());
                Alpha_YTest = cat(2,Alpha_Y,SigAlpha*randn());
            else
                Alpha_XTest = cat(2,Alpha_X,0);
                Alpha_YTest = cat(2,Alpha_Y,0);
            end
            
            %dirrect sampling of allocations
            [ZTest]=Gibbs_Z(SMD,K+1,Mu_XTest,Mu_YTest,Alpha_XTest,Alpha_YTest);
                        
            %Prior Raio
            PR = Pk(K+1)/Pk(K);
            
            LAlloc_Current = p_Alloc(SMD,Mu_X,Mu_Y,Alpha_X,Alpha_Y,ones(1,K)/K);
            LAlloc_Test = p_Alloc(SMD,Mu_XTest,Mu_YTest,Alpha_XTest,Alpha_YTest,ones(1,K+1)/(K+1));
            AllocR = exp(LAlloc_Test-LAlloc_Current);
            
            %Posterior Ratio
            A = PR*AllocR/(Area*PDFgrid(ID));
            
            Accept = isinf(LAlloc_Current) & LAlloc_Current < 0;
            
            if rand<A || Accept
                Z=ZTest;
                K=K+1;
                Mu_X=Mu_XTest;
                Mu_Y=Mu_YTest;
                Alpha_X=Alpha_XTest;
                Alpha_Y=Alpha_YTest;
            end
            
            if nn>NBurnin %Then record in chain
                
                Chain(nn-NBurnin).N = K;
                Chain(nn-NBurnin).X = Mu_X';
                Chain(nn-NBurnin).Y = Mu_Y';
                Chain(nn-NBurnin).AlphaX = Alpha_X';
                Chain(nn-NBurnin).AlphaY = Alpha_Y';
                Chain(nn-NBurnin).ID = Z;
                
            end
            
        case 4  %Remove
            
            if K==1 %Then update chain and return
                if nn>NBurnin %Then record in chain
                    
                    Chain(nn-NBurnin).N = K;
                    Chain(nn-NBurnin).X = Mu_X';
                    Chain(nn-NBurnin).Y = Mu_Y';
                    Chain(nn-NBurnin).AlphaX = Alpha_X';
                    Chain(nn-NBurnin).AlphaY = Alpha_Y';
                    Chain(nn-NBurnin).ID = Z;
                    
                end
                continue;
            end
            
            %pick emitter to remove:
            ID =randi(K);
            
            Mu_XTest = Mu_X;
            Mu_YTest = Mu_Y;
            Alpha_XTest = Alpha_X;
            Alpha_YTest = Alpha_Y;
            
            %Remove from list
            Mu_XTest(ID) = [];
            Mu_YTest(ID) = [];
            Alpha_XTest(ID) = [];
            Alpha_YTest(ID) = [];
            
            %Gibbs allocation
            [ZTest]=Gibbs_Z(SMD,K-1,Mu_XTest,Mu_YTest,Alpha_XTest,Alpha_YTest);
            
            %Prior Raio
            PR = Pk(K-1)/Pk(K);
            
            %Probability Ratio of Proposed Allocation and Current Allocation 
            LAlloc_Current = p_Alloc(SMD,Mu_X,Mu_Y,Alpha_X,Alpha_Y,ones(1,K)/K);
            LAlloc_Test = p_Alloc(SMD,Mu_XTest,Mu_YTest,Alpha_XTest,Alpha_YTest,ones(1,K-1)/(K-1));
            AllocR = exp(LAlloc_Test-LAlloc_Current);
            
            %Posterior Ratio
            A = PR*AllocR;
            
            if rand<A
                Z=ZTest;
                K=K-1;
                Mu_X=Mu_XTest;
                Mu_Y=Mu_YTest;
                Alpha_X=Alpha_XTest;
                Alpha_Y=Alpha_YTest;
            end
            
            if nn>NBurnin %Then record in chain
                
                Chain(nn-NBurnin).N = K;
                Chain(nn-NBurnin).X = Mu_X';
                Chain(nn-NBurnin).Y = Mu_Y';
                Chain(nn-NBurnin).AlphaX = Alpha_X';
                Chain(nn-NBurnin).AlphaY = Alpha_Y';
                Chain(nn-NBurnin).ID = Z;
                
            end
            
    end    
    
    if DEBUG==1 %for testing
        figure(1111)
        axis equal
        scatter(SMD.X,SMD.Y,[],Z)
        hold on
        plot(Mu_X,Mu_Y,'ro','linewidth',4)
        legend(sprintf('Jump: %g',nn))
        xlabel('X(nm)')
        ylabel('Y(nm)')
        hold off
        pause(.001)
    elseif DEBUG == 2
        RadiusScale = 2;
        CircleRadius = sqrt((SMD.X_SE.^2 + SMD.Y_SE.^2) / 2) * RadiusScale;
        figure(1111)
        for oo = 1:max(Z)
            ID = Z==oo;
            Theta = linspace(0, 2*pi, 25)';
            Theta = repmat(Theta,[1,sum(ID)]);
            CircleX = repmat(CircleRadius(ID)',[25,1]).*cos(Theta) + repmat(SMD.X(ID)',[25,1]);
            CircleY = repmat(CircleRadius(ID)',[25,1]).*sin(Theta) + repmat(SMD.Y(ID)',[25,1]);
            A=plot(CircleX,CircleY);
            if oo == 1;hold on;end
            if ~isempty(A)
                set(A,'color',A(1).Color);
            end
        end
        plot(Mu_X,Mu_Y,'ro','linewidth',4)
        legend(sprintf('Jump: %g',nn))
        xlabel('X(nm)')
        ylabel('Y(nm)')
       
        hold off
        pause(.001)
    end
end

end

function [ZTest]=Gibbs_Z(SMD,K,Mu_X,Mu_Y,Alpha_X,Alpha_Y)
    %This function calculates updated allocations (Z)
    
    T=SMD.FrameNum;
    N=length(T);

    %nm1 = SMD.X - (Mu_X + Alpha_X.*SMD.FrameNum);
    %nm2 = SMD.Y - (Mu_Y + Alpha_Y.*SMD.FrameNum);

    %PX=normpdf(nm1,0,SMD.X_SE);
    %PY=normpdf(nm2,0,SMD.Y_SE);
    %Pold=PX.*PY+eps;
    %PNormOld=Pold./sum(Pold,2);

    % Will not match rounding of old method, but this will be more precise.
    %P = exp(-0.5 * ((nm1./SMD.X_SE).^2 + (nm2./SMD.Y_SE).^2));
    %P = P./((SMD.X_SE.*SMD.Y_SE).*(2*pi)) + eps;

    %P = normpdf2d( ...
    %    SMD.X, SMD.Y, ...
    %    Mu_X + Alpha_X.*SMD.FrameNum, Mu_Y + Alpha_Y.*SMD.FrameNum, ...
    %    SMD.X_SE, SMD.Y_SE ...
    %    ) + eps;
    P = allocProbs(SMD, Mu_X, Mu_Y, Alpha_X, Alpha_Y) + eps;

    if sum(sum(isnan(P)))
       [ZTest] = knnsearch([Mu_X',Mu_Y'],[SMD.X,SMD.Y]); 
    else 
        rng = rand(N,1);
        cs = (cumsum(P,2)./sum(P, 2))+eps;
        stencil = rng < cs;
        ZTest=K+1-sum(stencil,2);
    end

    if max(ZTest) > K
        fprintf("wtf");
    end

end

function [Mu,Alpha]=Gibbs_MuAlpha(ID,Z,X,T,Sigma,SigAlpha)
    %This function calculates updated Mu and Alpha (1D)
    
    if length(X)==1
        Mu=Gibbs_Mu(ID,Z,X,Sigma);
        Alpha = 0;
        return;
    end
    
    if sum(Z==ID)==0
        Mu = X(randi(length(X)));
        Alpha = -SigAlpha+2*SigAlpha*rand();
    else
        %Get the localizations from the IDth emitter
        Xs=X(Z==ID);
        Sigs = Sigma(Z==ID);
        Ts=T(Z==ID);

        A = sum(Sigs.^-2);
        B = sum(Ts./Sigs.^2);
        D = sum((Ts.^2)./(Sigs.^2));

        %MLE estimates of Mu and Alpha

        [Alpha,Center] = calAlpha(Xs,Sigs,Ts,SigAlpha);
        MA=[Center;Alpha];

        %Covariance matrix Sigma
        COV = pinv([A, B;B,D+1/SigAlpha^2]);

        %This draws [Mu,Alpha] from a multivariate normal
        MuAlpha=mvnrnd(MA,COV);
        Mu=MuAlpha(1);
        Alpha=MuAlpha(2);

        if Mu == Center
           Mu = Center + sqrt(A)*randn(); 
        end
    end
    
end

function [Mu_X, Mu_Y]=Gibbs_Mu_Bulk(N, Z, X, Y, Sigma_X, Sigma_Y)
    Ax = accumarray(Z, X ./ (Sigma_X.^2), [N 1], @sum, single(-1));
    Ay = accumarray(Z, Y ./ (Sigma_Y.^2), [N 1], @sum, single(-1));

    Bx = accumarray(Z, Sigma_X.^-2, [N 1], @sum, single(-1));
    By = accumarray(Z, Sigma_Y.^-2, [N, 1], @sum, single(-1));

    XMLE = Ax./Bx;
    YMLE = Ay./By;
    X_SE = 1./sqrt(Bx);
    Y_SE = 1./sqrt(By);

    Mu_X = randn(size(XMLE), 'like', XMLE) .* X_SE + XMLE;
    Mu_Y = randn(size(YMLE), 'like', YMLE) .* Y_SE + YMLE;

    % Choose a random X,Y to act as Mu if there are no localizations for an
    % emitter
    no_locs = Ax==-1;
    Mu_X(no_locs) = X(randi(length(Z), sum(no_locs)));
    Mu_Y(no_locs) = Y(randi(length(Z), sum(no_locs)));
end

function [Mu]=Gibbs_Mu(ID,Z,X,Sigma)
    %This function calculates updated Mu (1D)

    assigned_locs = Z==ID;
    
    %Get the localizations from the IDth emitter
    if sum(assigned_locs) == 0
        Mu = X(randi(length(Z)));
    else
        Xs=X(assigned_locs);
        Sigs = Sigma(assigned_locs);
        A = sum(Xs./(Sigs.^2));
        B = sum(Sigs.^-2);
        XMLE = A/B;
        X_SE = 1/sqrt(B);
        %Mu=normrnd(XMLE,X_SE);
        Mu = randn(size(XMLE), 'like', XMLE) .* X_SE + XMLE;
    end
end

function [Alpha,Center] = calAlpha(Xs,Sigs,Frames,SigAlpha)
    Frames = single(Frames);
    A = sum(Xs./Sigs.^2);
    B = sum(Frames./Sigs.^2);
    C = sum(Sigs.^-2);
    AlphaTop = sum((C*Xs-A).*Frames./Sigs.^2);
    AlphaBottom = sum((C*Frames-B).*Frames./Sigs.^2)+C/SigAlpha^2;
    Alpha = AlphaTop/AlphaBottom;
    Center = (A-Alpha*B)/C;
end

function LogL = p_Alloc(SMD,Mu_X,Mu_Y,Alpha_X,Alpha_Y,Ws)
%This function calculated the probability of a given allocation set.
    %pdfs = Ws.*normpdf2d( ...
    %    SMD.X, SMD.Y, ...
    %    Mu_X + Alpha_X.*SMD.FrameNum, Mu_Y + Alpha_Y.*SMD.FrameNum, ...
    %    SMD.X_SE, SMD.Y_SE ...
    %    );
    pdfs = Ws.*allocProbs(SMD, Mu_X, Mu_Y, Alpha_X, Alpha_Y);

    LogL = log(sum(pdfs, 2));
    LogL = sum(LogL);
    
end

function P = allocProbs(SMD, Mu_X, Mu_Y, Alpha_X, Alpha_Y)
    Xgpu = gpuArray(SMD.X);
    Ygpu = gpuArray(SMD.Y);
    XSEgpu = gpuArray(SMD.X_SE);
    YSEgpu = gpuArray(SMD.Y_SE);
    Tgpu = gpuArray(SMD.FrameNum);
    MUXgpu = gpuArray(Mu_X);
    MUYgpu = gpuArray(Mu_Y);
    AXgpu = gpuArray(Alpha_X);
    AYgpu = gpuArray(Alpha_Y);
    
    Pgpu = normpdf2d(Xgpu, Ygpu, MUXgpu + AXgpu.*Tgpu, MUYgpu + AYgpu.*Tgpu, XSEgpu, YSEgpu);
    P = gather(Pgpu);
end

function P = normpdf2d(X, Y, Mu_X, Mu_Y, Sigma_X, Sigma_Y)
    % Based on the matlab mvnpdf implementation

    quadform = ((X - Mu_X) ./ Sigma_X).^2 + ((Y - Mu_Y) ./ Sigma_Y).^2;

    P = exp(-0.5*quadform - log(2*pi)) ./ (Sigma_X .* Sigma_Y);
end