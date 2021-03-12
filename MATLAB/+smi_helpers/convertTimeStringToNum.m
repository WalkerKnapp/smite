function [TimeNum] = convertTimeStringToNum(TimeString, Delimiter)
%convertTimeStringToNum converts a time string to a number.
% This function will take a timestring (i.e., the type output from
% genTimeString()) and convert it to a number.
%
% INPUTS: 
%   TimeString: A time string given in the format output by genTimeString()
%               NOTE: Changing the input 'MinFieldWidth' of genTimeString()
%                     might cause issues in this function!
%   Delimiter: A delimiter between the year, month, day, etc. in TimeString
%              (Default = '_')
%
% OUTPUTS:
%   TimeNum: A number corresponding to TimeString.
%            CAUTION: I haven't been careful about this conversion, so use
%                     caution if trying to use this number for anything
%                     other than, e.g., file sorting, where the absolute
%                     time doesn't really matter.

% Created by:
%   David J. Schodt (Lidke Lab, 2021)


% Set defaults if needed.
if (~exist('Delimiter', 'var') || isempty(Delimiter))
    Delimiter = '_';
end

% Break up the time string and convert to a number.
SplitTimeStamp = strsplit(TimeString, Delimiter);
for jj = 1:numel(SplitTimeStamp)
    % Ensure we pad with zeros wherever needed, e.g., 2018-2-9-17-41
    % should be 2018-02-09-17-41
    if mod(numel(SplitTimeStamp{jj}), 2)
        SplitTimeStamp{jj} = ['0', SplitTimeStamp{jj}];
    end
end

% Convert the cell array to a double.
TimeNum = str2double(cell2mat(SplitTimeStamp));


end