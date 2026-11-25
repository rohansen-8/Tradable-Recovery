function fileName = save_model_state(cfg, model_state)
%% Save model_state fields to cfg.dataset.modelStateFile.

    if ~isfolder(cfg.dataset.path)
        mkdir(cfg.dataset.path);
    end

    fileName = cfg.dataset.modelStateFile;
    if isfield(cfg.training,'save_model_state_as_variable') ...
            && cfg.training.save_model_state_as_variable
        save(fileName,'model_state','-v7.3');
    else
        save(fileName,'-struct','model_state','-v7.3');
    end
end
