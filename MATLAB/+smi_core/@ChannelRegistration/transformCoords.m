function [MovingCoordinates, FixedCoordinates] = transformCoords(...
    RegistrationTransform, MovingCoordinates, FixedCoordinates)
%transformCoords transforms a set of coordinates with the given transform.
% This method transforms the input coordinates, with the set being
% transformed depending on the type of transform.  The purpose of this
% method is to eliminate the guesswork needed when applying the transform
% (e.g., 'lwm' doesn't have a forward transform, so it has to be applied to
% the fixed coordinates to do what we want it to).
%
% INPUTS:
%   RegistrationTransform: RegistrationTransform is a MATLAB tform object
%                          describing the transform to be visualized.
%                          (tform object)
%   MovingCoordinates: Coordinates of points in the second fiducial.
%                      (Nx2 numeric array)
%   FixedCoordinates: Coordinates of points in the first fiducial.
%                     (Nx2 numeric array)
%
% OUTPUTS:
%   MovingCoordinates: Coordinates of points in the second fiducial.  These
%                      coordinates may or may not be transformed, depending
%                      on the type of transform used.
%                      (Nx2 numeric array)
%   FixedCoordinates: Coordinates of points in the first fiducial.  These
%                     coordinates may or may not be transformed, depending
%                     on the type of transform used.
%                     (Nx2 numeric array)

% Created by:
%   David J. Schodt (Lidke Lab, 2020)


% Determine the transformation type and proceed as appropriate.
if ismember(class(RegistrationTransform), ...
        {'images.geotrans.LocalWeightedMeanTransformation2D', ...
        'images.geotrans.PolynomialTransformation2D', ...
        'images.geotrans.PiecewiseLinearTransformation2D'})
    % None of these transforms have a transformPointsForward()
    % implementation, so we must use the inverse on our reference points
    % (We could just compute the transform treating the reference as
    % moving, but then the transform can't be applied to our raw data since
    % MATLAB doesn't have an inverse of imwarp().  For SPT, it's nice to
    % have transformed raw data for use in movies.)
    % NOTE: Since the moving images are transformed forward, but fixed
    %       coordinates are transformed in the inverse, we get an 
    %       additional offset that we need to subtract!
    FixedCoordinatesTrans = ...
        smi_core.ChannelRegistration.transformCoordsDirect(...
        RegistrationTransform, FixedCoordinates);
    TempMoving = ...
        smi_core.ChannelRegistration.transformCoordsDirect(...
        RegistrationTransform, MovingCoordinates);
    Offset = TempMoving - MovingCoordinates;
    FixedCoordinates = FixedCoordinatesTrans - Offset;
    MovingCoordinates = MovingCoordinates - Offset;
else
    MovingCoordinates = ...
        smi_core.ChannelRegistration.transformCoordsDirect(...
        RegistrationTransform, MovingCoordinates);
end


end