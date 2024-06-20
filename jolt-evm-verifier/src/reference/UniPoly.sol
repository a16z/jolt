// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Fr} from "./Fr.sol";

struct UniPoly {
    Fr[] coeffs;
}

library UniPolyLib {

    // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
    // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
    function decompress(Fr[] memory compressedCoeffs, Fr hint) internal pure returns (UniPoly memory){
        Fr linearTerm = hint - compressedCoeffs[0] - compressedCoeffs[0];
        Fr[] memory coeffs = new Fr[](compressedCoeffs.length + 1);
        for (uint i=1; i < compressedCoeffs.length; i++){
            linearTerm = linearTerm - compressedCoeffs[i];
        }

        coeffs[0] = compressedCoeffs[0];
        coeffs[1] = linearTerm;
        for (uint i = 1; i < compressedCoeffs.length; i++) {
            coeffs[i + 1] = compressedCoeffs[i];
        }

        return UniPoly(coeffs);

    }

    function evalAtOne(UniPoly memory poly) internal pure returns (Fr eval) {
        Fr sum = Fr.wrap(0);
        for (uint i = 0; i < poly.coeffs.length; i++) {
            sum = sum + poly.coeffs[i];
        }
        return sum;
    }

    function evalAtZero(UniPoly memory poly) internal pure returns (Fr eval){
        return poly.coeffs[0];
    }

    function evaluate(UniPoly memory poly, Fr r) internal pure returns (Fr) {
        Fr eval = poly.coeffs[0];
        Fr power = r;
        for (uint256 i=1; i < poly.coeffs.length; i++){
            eval = eval + (power * poly.coeffs[i]);
            power = power * r;
        }
        return eval;
    }

    

}
