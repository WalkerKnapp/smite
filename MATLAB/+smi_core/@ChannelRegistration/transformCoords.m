function [TransformedCoordinates] = transformCoords(...
    RegistrationTransform, Coordinates)
%transformCoords transforms a set of coordinates with the given transform.
% This method will transform the coordinates given in Coordinates using the
% transform in RegistrationTransform.  The way this is done will depend on
% the RegistrationTransform type (e.g., locally weighted mean transforms
% are applied differently than affine transforms).
%
% INPUTS:
%   RegistrationTransform: A MATLAB tform object containing information
%                          about the transformation to be used 
%                          (tform object)
%   Coordinates: Array of coordinates (Mx2 numeric array)
%
% OUTPUTS: 
%   TransformedCoordinates: Input Coordinates transformed using
%                           RegistrationTransform (Mx2 numeric array)

% Created by:
%   David J. Schodt (Lidke Lab, 2020)


% Determine the transformation type and proceed as appropriate.
if ismember(class(RegistrationTransform), ...
        {'images.geotrans.LocalWeightedMeanTransformation2D', ...
        'images.geotrans.PolynomialTransformation2D', ...
        'images.geotrans.PiecewiseLinearTransformation2D'})
    % None of these transforms have a transformPointsForward()
    % implementation, so we must use the inverse.
    [TransformedX, TransformedY] = transformPointsInverse(...
        RegistrationTransform, Coordinates(:, 1), Coordinates(:, 2));
    TransformedCoordinates = [TransformedX, TransformedY];
else
    [TransformedX, TransformedY] = transformPointsForward(...
        RegistrationTransform, Coordinates(:, 1), Coordinates(:, 2));
    TransformedCoordinates = [TransformedX, TransformedY];
end


end