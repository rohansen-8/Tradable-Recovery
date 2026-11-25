function projectRoot = get_project_root()
%% Root folder of the SDF project (the parent of configs/).
projectRoot = fileparts(fileparts(mfilename('fullpath')));
end
