function setpath()
%% Add project source folders to the MATLAB path.
rootDir = fileparts(fileparts(mfilename('fullpath')));

% Add source-code folders only.
% Generated data, grids, results, plots, and archives stay off the MATLAB path.
codeDirs = {'configs', 'drivers', 'kernels', 'training', ...
            'testing', 'plotting', 'evaluation', 'tests', ...
            fullfile('pipelines', 'data_construction'), ...
            fullfile('pipelines', 'empirical_oos_attribution'), ...
            fullfile('pipelines', 'economic_interpretation')};

for j = 1:numel(codeDirs)
    d = fullfile(rootDir, codeDirs{j});
    if exist(d, 'dir')
        addpath(d);
    end
end

% Simulator code is organized by model under simulators/.
% Add each simulator model folder explicitly.
simDir = fullfile(rootDir, 'simulators');
if exist(simDir, 'dir')
    addpath(simDir);
    sims = dir(simDir);
    for j = 1:numel(sims)
        if sims(j).isdir && ~startsWith(sims(j).name, '.')
            addpath(fullfile(simDir, sims(j).name));
        end
    end
end

end
