function assert_design_fingerprint(saved, expected)
%% Reject any checkpoint whose complete design fingerprint differs.

assert(isstruct(saved) && isscalar(saved), ...
    'Checkpoint is missing a scalar design fingerprint.');
assert(isequaln(saved, expected), ...
    ['Checkpoint design fingerprint mismatch. Refusing resume because the ' ...
     'saved output was produced under a different design.']);

end
