function x = ParseArguments(list, varargin)
% simple function to help in set defautl value for optional paramters of
% function. Could use inputParser instead in more recent Matlab versions.
for i = 1:2:numel(varargin)
    x = struct(varargin{:});
end
for i = 1:2:numel(list)
    x.(list{i}) = list{i+1};
end
