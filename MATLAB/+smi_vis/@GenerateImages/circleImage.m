function [CircleImage, CircleImageRGB, SRImageZoom] = ...
    circleImage(SMR, ColorMap, SRImageZoom, SEScaleFactor)
%circleImage generates an image with circles for each localization.
% This method generates a circle image of the localizations in SMR.  At
% each localization in SMR, a circle will be placed in the image centered
% at the localization coordinates, with the radius being the mean of X_SE
% and Y_SE for that localization.
%
% INPUTS:
%   SMR: Single Molecule Results structure (an SMD structure that may have
%        been "truncated" so that not all fields are present).
%   ColorMap: A colormap defining the colors of the circles.  The row
%             indices correspond to indices in SMR, i.e., the circle at
%             [SMR.X(ii), SMR.Y(ii)] will have color ColorMap(ii, :)
%             (numeric array, NLocalizations x 3, or NLocalizations x 1 for
%             a single color)(Default = [1, 0, 0])
%   SRImageZoom: Zoom factor of the output images w.r.t. the coordinate
%                system of the localizations.
%                (scalar, integer)(Default = 20)
%   SEScaleFactor: Multiplicative scaling of the mean X/Y standard errors.
%                  This is useful if you have very low SEs but still want
%                  their circles visible, without the need to increase
%                  SRImageZoom. (Default = 1);
%
%
% OUTPUTS:
%   CircleImage: A grayscale image with localizations in SMR represented by
%                circles, with radii varying with the standard errors.
%                (single array)(dynamic range of [0, 1])
%   CircleImageRGB: Same as output CircleImage but with RGB color channels.
%                   The color will be defined by the input ColorMap.
%                   (single array, mxnx3)(dynamic range of [0, 1])
%   SRImageZoom: Same as input 'SRImageZoom' unless the requested zoom
%                produces an image too large to be written with imwrite().
%                In that case, this will be the value which was actually
%                used.

% Created by:
%   David J. Schodt (Lidke lab, 2021)


% Define default parameters.
if (~exist('ColorMap', 'var') || isempty(ColorMap))
    ColorMap = [1, 0, 0];
end
if (~exist('SRImageZoom', 'var') || isempty(SRImageZoom))
    SRImageZoom = 20;
end
if (~exist('SEScaleFactor', 'var') || isempty(SEScaleFactor))
    SEScaleFactor = 1;
end

% Make local copies of some SMR fields (for very large SMR structures this
% might save time over accessing the fields within the loop(s) below).
X = SMR.X;
X_SE = SMR.X_SE;
Y = SMR.Y;
Y_SE = SMR.Y_SE;

% Revise 'ColorMap' if the rows dimension isn't suitable (this will be used
% if the input is only one row, for monochrome images).
NLocalizations = numel(X);
if (size(ColorMap, 1) ~= NLocalizations)
    ColorMap = repmat(ColorMap(1, 1:3), NLocalizations, 3);
end

% Define some new parameters.
ImageSize = SRImageZoom * [SMR.YSize, SMR.XSize];
RGBChannels = 3;
if ((RGBChannels*prod(ImageSize)) > (2^32 - 1))
    % imwrite() can't write images this large, so we'll choose a smaller
    % image size.
    ImageSize = floor([1, 1] * 2^16 / sqrt(3));
    RequestedSRImageZoom = SRImageZoom;
    SRImageZoom = ImageSize(1) / SMR.YSize;
    warning(['smi_vis.GenerateImages.circleImage(): ', ...
        'Requested SRImageZoom=%i is too large and was reset to ', ...
        'SRImageZoom=%i'], RequestedSRImageZoom, SRImageZoom)
end

% Rescale the data based on SRImageZoom.
X = X * SRImageZoom;
X_SE = X_SE * SRImageZoom;
Y = Y * SRImageZoom;
Y_SE = Y_SE * SRImageZoom;

% Generate the monochrome circle image.
% NOTE: The monochrome circle image SRImageZoom is restricted to the same
%       value as the RGB image (for the sake of simplicity in this code).
MeanSE = mean([X_SE, Y_SE], 2) * SEScaleFactor;
CircleImage = zeros(ImageSize, 'single');
for nn = 1:NLocalizations
    % Create the binary circle image by creating ~4*circumference
    % points for each circle (the extra points help make the circle
    % appear smooth).
    Theta = linspace(0, 2*pi, ceil(8*pi*MeanSE(nn)));
    CircleRows = round(MeanSE(nn)*cos(Theta) + Y(nn));
    CircleCols = round(MeanSE(nn)*sin(Theta) + X(nn));
    IsValid = ((CircleRows<=ImageSize(1)) & (CircleRows>=1) ...
        & (CircleCols<=ImageSize(2)) & (CircleCols>=1));
    CircleRows = CircleRows(IsValid);
    CircleCols = CircleCols(IsValid);
    for mm = 1:numel(CircleRows)
        CircleImage(CircleRows(mm), CircleCols(mm)) = 1;
    end
end

% Generate the RGB image if needed.
if (nargout >= 2)
    CircleImageRGB = zeros([ImageSize, 3], 'single');
    for nn = 1:NLocalizations
        % Create the circles by with ~4*circumference points per circle
        % (the extra points can help make the circle appear smooth).
        Theta = linspace(0, 2*pi, ceil(4*pi*MeanSE(nn)));
        CircleRows = round(MeanSE(nn)*cos(Theta) + Y(nn));
        CircleCols = round(MeanSE(nn)*sin(Theta) + X(nn));
        IsValid = ((CircleRows<=ImageSize(1)) & (CircleRows>=1) ...
            & (CircleCols<=ImageSize(2)) & (CircleCols>=1));
        CircleRows = CircleRows(IsValid);
        CircleCols = CircleCols(IsValid);
        for mm = 1:numel(CircleRows)
            CircleImageRGB(CircleRows(mm), CircleCols(mm), 1) = ...
                ColorMap(nn, 1);
            CircleImageRGB(CircleRows(mm), CircleCols(mm), 2) = ...
                ColorMap(nn, 2);
            CircleImageRGB(CircleRows(mm), CircleCols(mm), 3) = ...
                ColorMap(nn, 3);
        end
    end
end


end