Copy of ark_ec::VariableBaseMSM with minor modifications to speed up known small element sized MSMs.

Modifications account for ~30% speed ups at 2^24 lookups, C=1, table size = 2^16.