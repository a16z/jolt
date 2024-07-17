pragma solidity ^0.8.21;

import {TestBase} from "./base/TestBase.sol";
import {HyperKZG, HyperKZGProof} from "../src/subprotocols/HyperKZG.sol";
import {FiatShamirTranscript, Transcript} from "../src/subprotocols/FiatShamirTranscript.sol";

import "forge-std/console.sol";

struct VK {
    uint256 VK_g1_x;
    uint256 VK_g1_y;
    uint256[] VK_g2;
    uint256[] VK_beta_g2;
}

// Need to have a version which inits the immutables
contract DeployableHyperKZG is HyperKZG {
    constructor(VK memory vk) {
        VK_g1_x = vk.VK_g1_x;
        VK_g1_y = vk.VK_g1_y;
        VK_g2_x_c0 = vk.VK_g2[0];
        VK_g2_x_c1 = vk.VK_g2[1];
        VK_g2_y_c0 = vk.VK_g2[2];
        VK_g2_y_c1 = vk.VK_g2[3];
        VK_beta_g2_x_c0 = vk.VK_beta_g2[0];
        VK_beta_g2_x_c1 = vk.VK_beta_g2[1];
        VK_beta_g2_y_c0 = vk.VK_beta_g2[2];
        VK_beta_g2_y_c1 = vk.VK_beta_g2[3];
    }
}

contract TestHyperKZG is TestBase {
    struct Example {
        VK vk;
        HyperKZGProof proof;
        uint256 commitment_x;
        uint256 commitment_y;
        uint256[] point;
        uint256 claim;
    }

    struct BatchedExample {
        VK vk;
        HyperKZGProof proof;
        uint256[] commitments;
        uint256[] point;
        uint256[] claims;
    }

    function testHyperKZGPasses() public {
        // Invoke the rust to get a non trivial example proof
        string[] memory cmds = new string[](1);
        cmds[0] = "./script/target/release/hyperkzg_example";
        bytes memory result = vm.ffi(cmds);
        Example memory data = abi.decode(result, (Example));
        // Now deploy a verifier with the key inited
        HyperKZG verifier = new DeployableHyperKZG(data.vk);
        // We build a transcript in memory
        bytes32 start_string = "TestEval";
        Transcript memory transcript = FiatShamirTranscript.new_transcript(start_string, 3);
        // We call into the verifier contract
        bool passes =
            verifier.verify(data.commitment_x, data.commitment_y, data.point, data.claim, data.proof, transcript);
        require(passes, "does not verify a valid proof");
    }

    function testHyperKZGBatchPasses() public {
        // Invoke the rust to get a non trivial example proof
        string[] memory cmds = new string[](1);
        cmds[0] = "./script/target/release/hyperkzg_batch_example";
        bytes memory result = vm.ffi(cmds);
        BatchedExample memory data = abi.decode(result, (BatchedExample));
        // Now deploy a verifier with the key inited
        HyperKZG verifier = new DeployableHyperKZG(data.vk);
        // We build a transcript in memory
        bytes32 start_string = "TestEval";
        Transcript memory transcript = FiatShamirTranscript.new_transcript(start_string, 3);
        // We call into the verifier contract
        bool passes = verifier.batch_verify(data.commitments, data.point, data.claims, data.proof, transcript);
        require(passes, "does not verify a valid proof");
    }
}
