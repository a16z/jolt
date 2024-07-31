// SPDX-License-Identifier: MIT

pragma solidity >=0.8.0;

import {Transcript, FiatShamirTranscript} from "./FiatShamirTranscript.sol";
import {MODULUS, Fr, FrLib} from "./Fr.sol";

struct HyperKZGProof {
    uint256[] com; // G1 points represented pairwise
    uint256[] w; // G1 points represented pairwise
    uint256[] v_ypos; // Three vectors of scalars which must be ell length
    uint256[] v_yneg;
    uint256[] v_y;
}

// Implements a library to verify Hyperkzg opening proofs of commitments to multilinear polynomials
// Can't actually be a lib because we need immutables
contract HyperKZG {
    using FiatShamirTranscript for Transcript;
    using FrLib for Fr;

    // These are initialized using the trusted setup values. NOTE - You must negate the point
    // VK_g2 in order to have the checks pass, unlike the the rust which checks e(L, VK_G2) =
    // e(R, VK_Beta_G2) we use a precompile which checks e(L,-VK_G2)e(R, VK_Beta_G2) = 1
    uint256 immutable VK_g1_x;
    uint256 immutable VK_g1_y;
    // Note - These are elements of an elliptic curve over the extension field and so each has
    //        a c0 and c1 representing c0 + c1 X where X is the extending element
    uint256 immutable VK_g2_x_c0;
    uint256 immutable VK_g2_x_c1;
    uint256 immutable VK_g2_y_c0;
    uint256 immutable VK_g2_y_c1;
    uint256 immutable VK_beta_g2_x_c0;
    uint256 immutable VK_beta_g2_x_c1;
    uint256 immutable VK_beta_g2_y_c0;
    uint256 immutable VK_beta_g2_y_c1;

    /// Implements a batching protocol to verify multiple polynomial openings to the same point
    /// using hyper kzg and a random linear combination.
    /// @param commitments The polynomial commitment points in a vector arranged with x in the even
    ///                    and y in the odd positions
    /// @param point The point which is opened
    /// @param p_of_x The vector of claimed evaluations
    /// @param pi The proof of the opening, passed into the rlc verify
    /// @param transcript The fiat shamair transcript we are sourcing deterministic randoms from
    /// TODO - WARN - YOU MUST WRITE COMMITMENTS TO TRANSCRIPT BEFORE CALLING
    /// TODO - Affine and calldata pointer versions of this, to save gas on point rep
    function batch_verify(
        uint256[] memory commitments,
        uint256[] memory point,
        uint256[] memory p_of_x,
        HyperKZGProof memory pi,
        Transcript memory transcript
    ) public view returns (bool) {
        // Load a rho from transcript
        Fr rho = Fr.wrap(transcript.challenge_scalar(MODULUS));
        (uint256 running_x, uint256 running_y) = (commitments[0], commitments[1]);
        Fr running_eval = Fr.wrap(p_of_x[0]);
        Fr scalar = rho;
        for (uint256 i = 2; i < commitments.length; i += 2) {
            (uint256 next_x, uint256 next_y) = ec_scalar_mul(commitments[i], commitments[i + 1], scalar.unwrap());
            (running_x, running_y) = ec_add(running_x, running_y, next_x, next_y);
            running_eval = running_eval + Fr.wrap(mulmod(p_of_x[i / 2], scalar.unwrap(), MODULUS));
            scalar = scalar * rho;
        }
        // Pass the RLC into the singular verify function
        return (verify(running_x, running_y, point, running_eval.unwrap(), pi, transcript));
    }

    /// Implements the version multilinear hyper kzg verification as in the rust code at
    /// https://github.com/a16z/jolt/blob/main/jolt-core/src/poly/commitment/hyperkzg.rs
    /// @param c_x The x coordinate of the commitment to the multilinear polynomial
    /// @param c_y The y coordinate of the commitment to the multilinear polynomial
    /// @param point The point which is opened
    /// @param p_of_x The scalar which we are claiming is the evaluation of the polynomial
    /// @param pi The proof of the opening
    /// @param transcript The fiat shamair transcript we are sourcing deterministic randoms from
    function verify(
        uint256 c_x,
        uint256 c_y,
        uint256[] memory point,
        uint256 p_of_x,
        HyperKZGProof memory pi,
        Transcript memory transcript
    ) public view returns (bool) {
        // First append the points which are in the proof's com field
        transcript.append_points(pi.com);
        // Load a random from the transcript which is in the scalar field
        uint256 r = transcript.challenge_scalar(MODULUS);

        // No zeros on the points or random
        require(r != 0 && c_x != 0 && c_y != 0, "zeros");

        // now for the consistency checks
        uint256 ell = point.length;
        require(pi.v_y.length == ell && pi.v_yneg.length == ell && pi.v_ypos.length == ell, "bad length");

        for (uint256 i = 0; i < ell; i++) {
            uint256 y_i = i == ell - 1 ? p_of_x : pi.v_y[i + 1];
            Fr left = Fr.wrap(2) * Fr.wrap(r) * FrLib.from(y_i);
            Fr x_minus = FrLib.from(point[ell - i - 1]);
            Fr ypos_sub_yneg = FrLib.from(pi.v_ypos[i]) - FrLib.from(pi.v_yneg[i]);
            Fr ypos_plus_yneg = FrLib.from(pi.v_ypos[i]) + FrLib.from(pi.v_yneg[i]);
            // Get the other side of the equality
            Fr right = Fr.wrap(r) * (Fr.wrap(1) - x_minus) * ypos_plus_yneg + x_minus * ypos_sub_yneg;
            require(left == right, "bad construction");
        }

        // Now we proceed with a batched kzg verification
        // Unlike the rust code, we do not have v.com[0] = x,y or v.y[3] = p_of_x as we are avoiding
        // memory reallocations which would be necessary for pushing to the front or back of an array

        uint256 k = pi.com.length / 2 + 1;

        // Firstly we have to write our pi.v_y pi.v_yneg pi.v_ypos to the transcript, but because in rust
        // we do this with a rust array flatten we have to only write the append vector messages to the
        // transcript at the beginning and not in the middle
        transcript.append_bytes32("begin_append_vector");
        for (uint256 i = 0; i < pi.v_ypos.length; i++) {
            transcript.append_scalar(pi.v_ypos[i]);
        }
        for (uint256 i = 0; i < pi.v_yneg.length; i++) {
            transcript.append_scalar(pi.v_yneg[i]);
        }
        for (uint256 i = 0; i < pi.v_y.length; i++) {
            transcript.append_scalar(pi.v_y[i]);
        }
        // TODO - Check the rust code because the to_vec on p.v_y in the rust code clones and so it does not
        //        contain p_of_x in the final position like the Y vector created from pi.v[2].to_vec() does
        //        but the rust code is called using only p.v
        transcript.append_bytes32("end_append_vector");

        // Loads the q powers, which we do not have a separated function here so we do manually
        // later in this function (combining it with multiplying by the multiplier to save gas)
        Fr[] memory q_powers = new Fr[](k);
        Fr q = Fr.wrap(transcript.challenge_scalar(MODULUS));

        // Appends the points which are encoded in decompressed form.
        transcript.append_points(pi.w);
        // Loads the constants
        Fr d_0 = Fr.wrap(transcript.challenge_scalar(MODULUS));
        Fr d_1 = d_0 * d_0;
        // Must only have added three Ws to transcript
        require(pi.w.length == 6, "bad length");

        Fr q_power_multiplier = Fr.wrap(1) + d_0 + d_1;

        // Each element of q_powers is multiplier times q^i
        q_powers[0] = q_power_multiplier;
        for (uint256 i = 1; i < k; i++) {
            q_powers[i] = q_powers[i - 1] * q;
        }

        // Now calculate the B_u which is three positions of the random linear combination of the elle elements
        // of the v vectors in the proof.
        Fr B_u_ypos = Fr.wrap(0);
        Fr B_u_y = Fr.wrap(0);
        Fr B_u_yneg = Fr.wrap(0);
        Fr accumulated_q = Fr.wrap(1);
        for (uint256 i = 0; i < pi.v_ypos.length; i++) {
            B_u_ypos = B_u_ypos + FrLib.from(pi.v_ypos[i]) * accumulated_q;
            B_u_yneg = B_u_yneg + FrLib.from(pi.v_yneg[i]) * accumulated_q;
            B_u_y = B_u_y + FrLib.from(pi.v_y[i]) * accumulated_q;
            accumulated_q = q * accumulated_q;
        }

        // Finally we do a MSM to get the value of the the left hand side
        // NOTE - This is gas inefficient and grows with log of the proof size so we might want
        //        to move to a pippenger window algo with much smaller MSMs which we might save gas on.
        // Our first value is the c_x c_y as this would be the first entry of com in rust.
        (uint256 L_x, uint256 L_y) = ec_scalar_mul(c_x, c_y, q_powers[0].unwrap());

        // Now we do a running sum over the points in com
        for (uint256 i = 0; i < pi.com.length; i += 2) {
            // First the scalar mult then the add
            (uint256 temp_x_loop, uint256 temp_y_loop) =
                ec_scalar_mul(pi.com[i], pi.com[i + 1], q_powers[i / 2 + 1].unwrap());
            (L_x, L_y) = ec_add(L_x, L_y, temp_x_loop, temp_y_loop);
        }

        // Next add in the W dot product U
        (uint256 temp_x, uint256 temp_y) = ec_scalar_mul(pi.w[0], pi.w[1], r);
        (L_x, L_y) = ec_add(L_x, L_y, temp_x, temp_y);
        // U[1] = -r * d_0
        (temp_x, temp_y) = ec_scalar_mul(pi.w[2], pi.w[3], mulmod(MODULUS - r, d_0.unwrap(), MODULUS));
        (L_x, L_y) = ec_add(L_x, L_y, temp_x, temp_y);
        // U[2] = r*r * d_1
        (temp_x, temp_y) = ec_scalar_mul(pi.w[4], pi.w[5], mulmod(mulmod(r, r, MODULUS), d_1.unwrap(), MODULUS));
        (L_x, L_y) = ec_add(L_x, L_y, temp_x, temp_y);
        // -(B_u[0] + d_0 * B_u[1] + d_1 * B_u[2])
        uint256 b_u = MODULUS - (B_u_ypos + d_0 * B_u_yneg + d_1 * B_u_y).unwrap();
        // Add in to the msm b_u Vk_g1
        (temp_x, temp_y) = ec_scalar_mul(VK_g1_x, VK_g1_y, b_u);
        (L_x, L_y) = ec_add(L_x, L_y, temp_x, temp_y);

        // Next we calculate the right hand side as the 3 part msm of the W values and the d_0 d_1 constants
        uint256 R_x = pi.w[0];
        uint256 R_y = pi.w[1];
        (temp_x, temp_y) = ec_scalar_mul(pi.w[2], pi.w[3], d_0.unwrap());
        (R_x, R_y) = ec_add(R_x, R_y, temp_x, temp_y);
        (temp_x, temp_y) = ec_scalar_mul(pi.w[4], pi.w[5], d_1.unwrap());
        (R_x, R_y) = ec_add(R_x, R_y, temp_x, temp_y);

        //Finally we check pairing(L, vk_g2) == pairing(R, vk_beta_g2)
        return (pairing(L_x, L_y, R_x, R_y));
    }

    /// Calculates nP where P is on the G1 curve of our ethereum precompile pairing
    /// Requires that the types be properly checked before calling
    /// @param p_x The x of the point Q
    /// @param p_y The y of the point Q
    /// @param n The scalar
    function ec_scalar_mul(uint256 p_x, uint256 p_y, uint256 n) internal view returns (uint256 x_new, uint256 y_new) {
        bool success;
        assembly ("memory-safe") {
            let prev_frm := mload(0x40)
            mstore(0, p_x)
            mstore(0x20, p_y)
            mstore(0x40, n)
            success := staticcall(gas(), 7, 0, 96, 0, 64)
            mstore(0x40, prev_frm)
            x_new := mload(0)
            y_new := mload(0x20)
        }
        require(success, "failing ec mul");
    }

    /// Calculates P + Q where P and Q are on the G1 curve of our ethereum precompile pairing
    /// Requires that the types be properly checked before calling
    /// @param p_x The x of the point P
    /// @param p_y The y of the point P
    /// @param q_x The x of the point P
    /// @param q_y The y of the point P
    function ec_add(uint256 p_x, uint256 p_y, uint256 q_x, uint256 q_y)
        internal
        view
        returns (uint256 x_new, uint256 y_new)
    {
        bool success;
        assembly ("memory-safe") {
            let prev_frm := mload(0x40)
            mstore(0, p_x)
            mstore(0x20, p_y)
            mstore(0x40, q_x)
            mstore(0x60, q_y)
            success := staticcall(gas(), 6, 0, 128, 0, 64)
            mstore(0x40, prev_frm)
            // do we really need this cleaning? if the compiler wants a zero value why would it ever mload?
            mstore(0x60, 0)
            x_new := mload(0)
            y_new := mload(0x20)
        }
        require(success, "failing ec add");
    }

    /// Checks that e(L, neg(VK_G1)) e(VK_G2, R) = 1. We must negate the G1 point in order
    /// to check e(L, VK_G1) = e(VK_G2, R) as the pairing precompile only checks equations
    /// Requires that the types be properly checked before calling
    /// @param L_x The x of the point L
    /// @param L_y The y of the point L
    /// @param R_x The x of the point R
    /// @param R_y The y of the point R
    function pairing(uint256 L_x, uint256 L_y, uint256 R_x, uint256 R_y) internal view returns (bool valid) {
        // put the immutables into local
        uint256 vk_g2_x_c0 = VK_g2_x_c0;
        uint256 vk_g2_x_c1 = VK_g2_x_c1;
        uint256 vk_g2_y_c0 = VK_g2_y_c0;
        uint256 vk_g2_y_c1 = VK_g2_y_c1;
        uint256 vk_g2_beta_x_c0 = VK_beta_g2_x_c0;
        uint256 vk_g2_beta_x_c1 = VK_beta_g2_x_c1;
        uint256 vk_g2_beta_y_c0 = VK_beta_g2_y_c0;
        uint256 vk_g2_beta_y_c1 = VK_beta_g2_y_c1;
        // Now run the pairing check in assembly
        bool success;
        assembly ("memory-safe") {
            // Here we have too many slots to fit into the first 4 scratch space slots and so we write after the
            // frm, which is still memory safe as high level sol doesn't assume cleaning and neither do we.
            let ptr := mload(0x40)
            // G1 point L
            mstore(ptr, L_x)
            ptr := add(ptr, 0x20)
            mstore(ptr, L_y)
            ptr := add(ptr, 0x20)
            // G2 point -VK.G2 note, we assume that this is initialized to it's negative
            mstore(ptr, vk_g2_x_c1)
            ptr := add(ptr, 0x20)
            mstore(ptr, vk_g2_x_c0)
            ptr := add(ptr, 0x20)
            mstore(ptr, vk_g2_y_c1)
            ptr := add(ptr, 0x20)
            mstore(ptr, vk_g2_y_c0)
            ptr := add(ptr, 0x20)
            // G1 point R
            mstore(ptr, R_x)
            ptr := add(ptr, 0x20)
            mstore(ptr, R_y)
            ptr := add(ptr, 0x20)
            // G2 point VK.beta_G2
            mstore(ptr, vk_g2_beta_x_c1)
            ptr := add(ptr, 0x20)
            mstore(ptr, vk_g2_beta_x_c0)
            ptr := add(ptr, 0x20)
            mstore(ptr, vk_g2_beta_y_c1)
            ptr := add(ptr, 0x20)
            mstore(ptr, vk_g2_beta_y_c0)
            ptr := add(ptr, 0x20)
            success := staticcall(gas(), 8, mload(0x40), 384, ptr, 32)
            valid := mload(ptr)
        }
        require(success, "failing pairing");
    }
}
