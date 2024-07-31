// SPDX-License-Identifier: MIT
pragma solidity >=0.8.21;

import {Transcript, FiatShamirTranscript} from "./FiatShamirTranscript.sol";
import {MODULUS, Fr, FrLib} from "./Fr.sol";

import "forge-std/console.sol";

struct SumcheckInstanceProof {
    uint256[][] compressedPolys;
}

error SumcheckFailed();
error InvalidLength();
error BadDegree();

library SumcheckVerifier {
    using FiatShamirTranscript for Transcript;
    using FrLib for Fr;

    /// Verifies the sumcheck component pre opening proof. The opening proof must be done
    /// later. Since the inputs are not trusted we don't use Fr and instead mod all inputs.
    /// @param transcript The running fiat shamir transcript
    /// @param proof The sumcheck proof
    /// @param claim The claimed sum
    /// @param num_rounds The depth of the sumcheck
    /// @param degree The degree of the interpolating univariate polynomials
    function verify_sumcheck(
        Transcript memory transcript,
        SumcheckInstanceProof memory proof,
        Fr claim,
        uint256 num_rounds,
        uint256 degree
    ) internal pure returns (Fr, Fr[] memory) {
        if (proof.compressedPolys.length != num_rounds || degree > 3) {
            revert InvalidLength();
        }

        Fr e = claim;
        Fr[] memory r = new Fr[](num_rounds);

        for (uint256 i = 0; i < num_rounds; i++) {
            //verify degree bound
            if (proof.compressedPolys[i].length != degree) {
                revert BadDegree();
            }

            // TODO - We can move this into the transcript lib
            transcript.append_bytes32("UniPoly_begin");
            for (uint256 j = 0; j < proof.compressedPolys[i].length; j++) {
                // We have to mod because the FR type cannot be used for untrusted inputs
                transcript.append_scalar(proof.compressedPolys[i][j] % MODULUS);
            }
            transcript.append_bytes32("UniPoly_end");

            // Sample random from the transcript
            Fr r_i = Fr.wrap(transcript.challenge_scalar(MODULUS));
            r[i] = r_i;

            e = evaluateCompressed(proof.compressedPolys[i], e, r_i);
        }
        return (e, r);
    }

    /// We evaluate a compressed poly at a point by uncompressing the linear term the going term wise
    /// @param compressedCoeffs The compressed coefficients of the poly
    /// @param hint The hint to help recover the linear term
    /// @param point The point we evaluate at
    function evaluateCompressed(uint256[] memory compressedCoeffs, Fr hint, Fr point)
        internal
        pure
        returns (Fr linear)
    {
        // Calculate the implied linear term
        // We are anti jumpi and try to minimize them
        Fr c0 = FrLib.from(compressedCoeffs[0]);
        Fr c2 = FrLib.from(compressedCoeffs[1]);
        Fr c3 = (compressedCoeffs.length == 2) ? Fr.wrap(0) : FrLib.from(compressedCoeffs[2]);
        Fr c1 = hint - Fr.wrap(2) * c0 - c2 - c3;
        // Evaluate
        Fr x = point;
        Fr eval = c0 + c1 * x;
        x = x * x;
        return (eval + c2 * x + c3 * x * point);
    }
}
