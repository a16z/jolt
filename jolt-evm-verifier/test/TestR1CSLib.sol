pragma solidity ^0.8.21;

import {TestBase} from "./base/TestBase.sol";
import {R1CSMatrix} from "../src/subprotocols/R1CSMatrix.sol";
import {Fr, FrLib} from "../src/reference/Fr.sol";

import "forge-std/console.sol";


contract TestR1CSLib is TestBase {
    using FrLib for Fr;

    // Here we are not calling out directly to rust because these functions are sometimes internal
    // instead we load examples from the sha2 test chain and compare them. Integration tests are
    // implied by integration tests of the full spartan R1CS test.

    function testEQPolyEvaluate() public pure {
        // Define two test vectors
        uint256[] memory r_u = new uint256[](19);
        r_u[0] = 0x196aff7ae9bf57ac8253c1ec87373875a7108a267b2efd3b67b74053c0143200;
        r_u[1] = 0x09588c034be7e0930344e7678b7bc3be84a76ff1fb31ca21516e551f1fc6e0f9;
        r_u[2] = 0x11dce9fc1c842ed684eae7906d38abb48bad91af4b79909cdb10882ac8e9f5ba;
        r_u[3] = 0x0e787025a753573fd42c1e2d54016379fbb2654bbda2a673587e2c6bfe8325ac;
        r_u[4] = 0x0853a26bfe80fd85a818557046f25a0d73bd95f00a4dfb8f73667720b06ac952;
        r_u[5] = 0x160e38bebb4bed4f5d6ccd46e5a636f32987cc0c203e85e077f14d35eb3c6852;
        r_u[6] = 0x1ae861092a04c4672d1057f650fda4a5e588d1e15c0272d75788c8b129490ad1;
        r_u[7] = 0x01f3e3d33679888db3a1bfb7b5acb39bc5e026da42888dedb545186fae2ab9a9;
        r_u[8] = 0x241b40362e508e32ce21b3c79a88ea8fd168598d58dee4a81b21dc12a32686b9;
        r_u[9] = 0x1de0c3acd309ec54c68cdb59ae33e953604b572334ef192959956040d5253e8a;
        r_u[10] = 0x24db1db6cdf238f959f32d4f67a1661649cf7ee7d58db17d13e9eb5a7b0f43cd;
        r_u[11] = 0x0f8c1f32675d23b47a735c618a73016f056a49f765893047a7f7b868a7b3f02b;
        r_u[12] = 0x20a74ba8aabb4978d58b0b64e84ad3a665c8e402200be6441e228e6760f1578a;
        r_u[13] = 0x2a75b2771dabcbc550aceac510a87dd09479daf98650e43a04117e9d9a38a134;
        r_u[14] = 0x1c831d026ee123647ce589889cfbff8e602efd5c3be00345e12f4215225de6be;
        r_u[15] = 0x1e54f4b8b27b7a12a48963e85f97fa62ad0463c3b8e5d9957b5cddee34d8a387;
        r_u[16] = 0x292afca29461e56e6a5872c40f7b11feb2ba1188bcc791b09b5366e1e5f27025;
        r_u[17] = 0x14a7a340e3cd047a608753d77c3018664b32fbd6376b764c08f3da5282f1c78c;
        r_u[18] = 0x15f6161c948fd0d95bd527e9ee90907d2b26ce4a607093dc52dd739abc2cc8e3;
        uint256[] memory e_u = new uint256[](19);
        e_u[0] = 0x05be3b5c439206998e9b17da2a58e754ad47d932dde3948dd2b38e39241c5dc3;
        e_u[1] = 0x01b0a65284b5a826320be7b2e87e50092944728c976c999c67d05554ef5edac5;
        e_u[2] = 0x2e0e5eb876de215f02af3602694002edb3e38eaef26391fb3f48a4815ad0ab51;
        e_u[3] = 0x0a56a873f2d5e01f7d9008c1c7725d3f28ae7cd33c18a46ddc8e99a5303d4aa8;
        e_u[4] = 0x033d4c6def35c5004179aae1cbc1f1e83d0c14684ab86c0bcfcd1827b9196dd8;
        e_u[5] = 0x2538bdaeeea948674dedc1825f4766278992b2d6fad789bb6df6815e18f65879;
        e_u[6] = 0x1b7513a75550eb5c6de6c541e5691f46a0c3686f7a8298ddf4861173c5a9bfd0;
        e_u[7] = 0x1d352daf0bf49581e105bd4e92244d318ad9437073a83ae5e1a8017cdaafc986;
        e_u[8] = 0x09d6b7d7faf3782abdf5a5717cfcab53e8169459932ed5d89266a28dda0e0fb0;
        e_u[9] = 0x1fcc0198e12b59b8ead649ce78d9a4299abd86aa0ec2478d45b1ec74c9b2bd4e;
        e_u[10] = 0x233a1ed329335c42d1a67b01ab4805cb2445ecbcc436bfadb5074cb1b21e07b0;
        e_u[11] = 0x2bb6e561b22b37bbb0837d294d6a42d040256df6253455bef96cd927060fb883;
        e_u[12] = 0x1356ce62cfd76c3464508ee2ae8a6375f9691d8c8f9a88b9ff2d28fe731dc98b;
        e_u[13] = 0x1e9ceeb169d76cab6a2ee528202625800388ea00ceb17920a3b259c0692fe466;
        e_u[14] = 0x1956d271cae0ee1852ef80b1b412f9e801129c3e5070816db32f6e6d8f1e6f43;
        e_u[15] = 0x20c1ccb5af9409f762fca6756a2775b0d8eed44ad75b17843dd976bfa41bea42;
        e_u[16] = 0x0b037113a1e9416d0408695deb31e8f55a74e95b8734257c9811e0c620412683;
        e_u[17] = 0x0d009eef88e21bc3bec8153a6420788fe257da1fec36ea30e268b32a1264993e;
        e_u[18] = 0x272e4b634b09a092fef121e091c884bb25540ff00c3317133a0eb5ba1ecf0b96;
        Fr[] memory r;
        Fr[] memory e;
        assembly {
            r := r_u
            e := e_u
        }
        // Now run the eq poly function to evaluate EqPoly(r).evaluate(e)
        Fr res = R1CSMatrix.eq_poly_evaluate(r, 0, 19, e, 0, 19);
        assertEq(res.unwrap(), 0x00bfae5e98717dcb1a3bcbc7c6d0fb4948856df8fd0a307e99f5596ea9f87a24);
    }
}