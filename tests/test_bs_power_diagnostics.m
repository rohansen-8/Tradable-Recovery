function test_bs_power_diagnostics()
%% Verify known BS population nulls and Mahalanobis cone diagnostics.

setpath();

horizons = ["quarterly"; "monthly"; "weekly"; "daily"];
regions = ["left"; "center"; "right"];
testTypes = ["mono"; "conv"; "conc"];

rows = cell(numel(horizons) * numel(regions) * numel(testTypes), 1);
k = 0;
for ih = 1:numel(horizons)
    cfg = config_bs_simulation_design(horizons(ih),'bs_empirical_t');
    for ir = 1:numel(regions)
        for it = 1:numel(testTypes)
            k = k + 1;
            rows{k} = table(ih, "test", "bs_empirical_t", horizons(ih), cfg.sim.empirical_T, ...
                1e-4, regions(ir), testTypes(it), 0.5, false, false, false, ...
                'VariableNames', {'design_id', 'design_group','experiment', 'horizon', 'Tval', ...
                'lambda', 'region', 'testType', 'p_value', ...
                'reject_1pct', 'reject_5pct', 'reject_10pct'});
        end
    end
end
raw = cat(1, rows{:});

[compact, population] = build_bs_power_diagnostics(raw, 40);

for ih = 1:numel(horizons)
    for ir = 1:numel(regions)
        h = horizons(ih);
        r = regions(ir);

        mono = population.horizon == h & population.region == r & population.testType == "mono";
        conv = population.horizon == h & population.region == r & population.testType == "conv";
        conc = population.horizon == h & population.region == r & population.testType == "conc";

        assert(population.true_null(mono));
        assert(population.true_null(conv));
        assert(~population.true_null(conc));
        assert(population.population_cone_distance(mono) == 0);
        assert(population.population_cone_distance(conv) == 0);
        assert(population.population_cone_distance(conc) > 0);
    end
end

assert(all(compact.diagnostic(compact.population_null == "true") == "size"));
assert(all(compact.diagnostic(compact.population_null == "false") == "power"));
assert(all(isfinite(compact.population_cone_distance)));

fprintf('test_bs_power_diagnostics passed.\n');

end
