function h = oos_sha256(x)
%% SHA-256 (lowercase hex) of a char/string (UTF-8) or a uint8 vector.
if ischar(x) || isstring(x)
    b = unicode2native(char(x), 'UTF-8');
else
    b = x;
end
assert(isa(b, 'uint8'), 'oos_sha256:Type', 'Input must be text or uint8.');
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(b(:));
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end
