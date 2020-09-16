function success = unitTest()
% Test Fourier Ring Correlation (FRC) interface functions and provide examples
% of usage.
%
% REQUIRES:
%    DIPlib Image Resolution add-on
%    Curve Fitting Toolbox (needed by qCorrectLocs)
%    Parallel Processing Toolbox
%    NVidia GPU
%
% CITATION:
%    http://www.diplib.org/add-ons
%    Image Resolution, Reference: R.P.J. Nieuwenhuizen, K.A. Lidke, M. Bates,
%    D. Leyton Puig, D. Grünwald, S. Stallinga, B. Rieger, Measuring Image
%    Resolution in Optical Nanoscopy, Nature Methods, 10(6):557-562, 2013.
% NOTE:
%    Install the Image Resolution software at the same level as sma-core-alpha.
%    This software is located at the URL above (see CITATION).  In startup.m,
%    add a path to
%       .../FRCresolution_software/matlabdistribution/FRCresolutionfunctions
%    where often ... = /Documents/MATLAB.  In the FRCresolutionfunctions, copy
%    smooth.m from the MATLAB Curve Fitting Toolbox into cfsmooth.m .  For
%    example, look in
%       MATLAB_DISTRIBUTION/toolbox/curvefit/curvefit/smooth.m
%    This is needed because DIPimage also has a smooth function which will
%    typically shadow MATLAB's smooth.

   PixelSize = 100;   % nm
   SRZoom = 10;

   shape = 'star';    % pattern type
   rho = 100;         % fluorophore density (fluorphore/pixel)
   XYSize = 64;       % linear size of image (pixels)
   n_frames = 1000;   % number of frames

   % Generate localizations and put them into an SMR structure.
   [Data, SMR] = SMA_Sim.smlmData(shape, 'No', 1, 0.0005, 1, 2000, rho, ...
                                  1.3, 15, XYSize, n_frames, 'Equib');

   n_particles = numel(SMR.X);
   SMR.XSize = XYSize;
   SMR.YSize = XYSize;
   SMR.X_SE = (PixelSize/1000)*ones(n_particles, 1);
   SMR.Y_SE = (PixelSize/1000)*ones(n_particles, 1);
   SMR.Ndatasets = 1;
   SMR.DatasetNum = ones(n_particles, 1);
   SMR.NFrames = max(SMR.FrameNum);

   %% ---------- uncorrected resolution calculation

   FRCc = smi_core.FRC();   % FRC class
   FRCc.PixelSize = PixelSize;
   FRCc.SRImageZoom = SRZoom;

   [res, ~, resH, resL] = FRCc.posToResolution(SMR);

   fprintf('resolution = %2.1f +- %2.2f [px]\n', ...
           res / SRZoom, (resL / SRZoom - resH / SRZoom)/2);
   fprintf('resolution = %2.1f +- %2.2f [nm]\n', ...
           res * PixelSize/SRZoom, (resL - resH)/2 * PixelSize/SRZoom);

   %% ---------- compute uncorrected FRC curve

   frc_curve = FRCc.posToFRC(SMR);
   [~, frc_curveA] = FRCc.posToResolution(SMR);

   figure();
   hold on
   qmax = 0.5 / (PixelSize/SRZoom);
   plot(linspace(0, qmax*sqrt(2), numel(frc_curve)), frc_curve,  'b-');
   plot(linspace(0, qmax*sqrt(2), numel(frc_curve)), frc_curveA, 'g-');
   xlim([0, qmax]);
   plot([0, qmax], [1/7, 1/7], 'r-');
   plot([0, qmax], [0, 0], 'k--');
   xlabel('spatial frequency (nm^{-1})');
   ylabel('FRC');
   title('Fourier Ring Correlation curve');
   legend({'posToFRC', 'posToResolution'}, 'Location', 'NorthEast');
   hold off

   %% ---------- compute uncorrected and corrected FRC curves and resolutions
   % Correction removes spurious correlations and is the recommended method to
   % use for typical applications.

   [res_corr, res_uncorr, Q, frc_curve_corr, frc_curve_uncorr] = ...
      FRCc.qCorrectionLocs(SMR);

   res1 = frctoresolution(frc_curve_uncorr, SMR.YSize*SRZoom);
   res2 = frctoresolution(frc_curve_corr,   SMR.YSize*SRZoom);
   fprintf('resolution = %2.1f, corrected = %2.1f [px], Q = %f\n', ...
           res_uncorr / SRZoom, res_corr / SRZoom, Q);
   fprintf('resolution = %2.1f, corrected = %2.1f [nm], Q = %f\n', ...
           res_uncorr * PixelSize/SRZoom, res_corr * PixelSize/SRZoom, Q);
   fprintf('frctoresolution = %2.1f, corrected = %2.1f [px]\n', res1, res2);

   figure();
   hold on
   qmax = 0.5 / (PixelSize/SRZoom);
   plot(linspace(0, qmax*sqrt(2), numel(frc_curve_corr)),   ...
        frc_curve_corr, 'b-');
   plot(linspace(0, qmax*sqrt(2), numel(frc_curve_uncorr)), ...
        frc_curve_uncorr, 'g-');
   xlim([0, qmax]);
   plot([0, qmax], [1/7, 1/7], 'r-');
   plot([0, qmax], [0, 0], 'k--');
   xlabel('spatial frequency (nm^{-1})');
   ylabel('FRC');
   title('Corrected Fourier Ring Correlation curve');
   legend({'corrected', 'uncorrected'}, 'Location', 'NorthEast');
   hold off

   success = 1;

end