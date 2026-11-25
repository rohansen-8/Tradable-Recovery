function test_bs_study_design()
%% Verify the bs_increasing_t and bs_empirical_t BS manifest.

setpath();
design=get_bs_study_design();
assert(height(design)==9);
assert(sum(design.design_group=="monthly_performance")==5);
assert(sum(design.design_group=="empirical_T")==4);
assert(isequal(design.Tval(design.design_group=="monthly_performance").', ...
    [500 1000 2000 5000 10000]));
emp=design(design.design_group=="empirical_T",:);
assert(isequal(emp.horizon,["quarterly";"monthly";"weekly";"daily"]));
assert(isequal(emp.Tval,[97;273;694;1964]));
assert(isequal(emp.option_panel_m_points,[46;77;27;28]));
assert(all(emp.option_panel_m_grid_type=="forward_moneyness_linspace"));
assert(all(cellfun(@(x)all(diff(x)>0),emp.option_m_grid)));
assert(all(design.n_m_grid==30));
% G differs by experiment: 52 for increasing-T, 60 for empirical-T.
assert(all(design.test_grid_n_points(design.design_group=="monthly_performance")==52));
assert(all(design.test_grid_n_points(design.design_group=="empirical_T")==60));
assert(all(design.inference_lambda_rule=="lambda_fit"));
assert(isequal(design.hac_bandwidth,[5;6;7;9;11;3;5;6;7]));

% Execute the exact manifest-column selection used by run_bs_study.
displayColumns=get_bs_display_columns();
displayDesign=design(:,displayColumns);
assert(isequal(displayDesign.Properties.VariableNames,displayColumns));
assert(height(displayDesign)==height(design));

end
