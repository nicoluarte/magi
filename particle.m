classdef particle
    properties
        position = [];
    end
    % constructor
    function obj = createPositionMatrix(obj, lower, upper, m, n)
        if nargin < 4
            lower = 0;
            upper = 10;
            m = 100;
            n = 2;
        end
        obj = unifrnd(lower, upper, m, n)
    end
end