    #[cfg(feature = "terminal-contraction")]
    #[inline(always)]
    pub fn acc_sub_out() {
        fixed!(r_word(FUNCT3_SUB, REG_ACC, REG_ACC, REG_OUT));
    }
