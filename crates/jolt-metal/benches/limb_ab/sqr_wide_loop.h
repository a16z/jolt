inline Words<8> sqr_wide(uint4 a) {
    Words<8> out;
    for (int k = 0; k < 8; k++) {
        out[k] = 0;
    }
    for (int i = 0; i < 3; i++) {
        ulong t = 0;
        for (int j = i + 1; j < 4; j++) {
            t = ulong(a[i]) * a[j] + out[i + j] + (t >> 32);
            out[i + j] = uint(t);
        }
        out[i + 4] = uint(t >> 32);
    }
    for (int k = 7; k > 0; k--) {
        out[k] = (out[k] << 1) | (out[k - 1] >> 31);
    }
    ulong t = 0;
    for (int i = 0; i < 4; i++) {
        ulong square = ulong(a[i]) * a[i];
        t = ulong(out[2 * i]) + uint(square) + (t >> 32);
        out[2 * i] = uint(t);
        t = ulong(out[2 * i + 1]) + uint(square >> 32) + (t >> 32);
        out[2 * i + 1] = uint(t);
    }
    return out;
}
