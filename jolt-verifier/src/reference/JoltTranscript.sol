// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {ITranscript} from "../interfaces/ITranscript.sol";
import {Fr} from "./Fr.sol";

contract JoltTranscript is ITranscript {

    //TODO: remove
    // mocking fiat shamir values 
    Fr[68] public challenges = [
        Fr.wrap(0x4121333f3452900d82acf910238dd7f9dcb1c8d96bd8438dc3161e01553c90e2),
        Fr.wrap(0x232b20a586978e89bd377e7f0543b01fec033b8f97e3f5bf21a937e6d7e62ef8),
        Fr.wrap(0x850b242c8452ed3c8ec15414dd7cff2e3f6fe5ed41b986f01e2bf4341ed15826),
        Fr.wrap(0x0dbd8a8160fc6d5190238e2af9b1d1ae1ce3144462a6c449b837852ca125c2bf),
        Fr.wrap(0xa5a4660193ae6e08330bbbff923f781b3bd67089a1957e2a7f7e56fd08a2a74d),
        Fr.wrap(0xf2a4bae23c05ebbd4479de37c691dbff70bc9e3b628450abe347567d79ad67a7),
        Fr.wrap(0x91792c5afdda68d1380077c3d9d4258442bb4ced671b65cb27fe7c6aa4206661),
        Fr.wrap(0xf7bce048f1058945ba57f3db9fc0c3834725a1998b1d577cc78f9305ded84c10),
        Fr.wrap(0xf67840be2d342641757c21cd63af6d55b92a753494e7e8f0739a8ecf71893326),
        Fr.wrap(0xc1b742a902e2f041b8406df6dc4b2d9e4a9922e4109cd07b2c3fd44e92878bb1),
        Fr.wrap(0x69cb9938ae8111f882e8f35e6c169578afb137ac3078f9990b268a8ab81b848a),
        Fr.wrap(0x67cb5d2c9fae93ae7889c61c38c56e353aa57faf1cbdf01c9bbaba12d43e3c22),
        Fr.wrap(0xd350de6c58501147c907c363d99df6dbcd2de4eb2356b241284f31ea6dbd5766),
        Fr.wrap(0x85f11dbddd8d8cf233850c61882c622c76c8b8a62493a90d787963a05efaa86d),
        Fr.wrap(0xe884dc9cb0ad7042ff033ae6a145bfcbaa29d2145df1a325ae8ac622d39aac35),
        Fr.wrap(0x1616e7696e1cbe462e119897628fe2b3fc9aab0ab22556aa48b1787d01c3b441),
        Fr.wrap(0x31ebbe36423ee0a0fa57abfedab7316fc2e4d2084b5d4dc23cc449aff0ebf840),
        Fr.wrap(0x3ee1fe5f650532d0b0d6e1cd3d2ed26db08be7dc754fbb1fad5c558cbacdcf51),
        Fr.wrap(0x01ffaddcabc7f0c624ed11c3881cef917a3b50c4f44815dddc2640c73761ea0b),
        Fr.wrap(0x70cabc670bc5869ac15dce40eaf3914e6ab92c96f5860ed38a002854ece3dfcf),
        Fr.wrap(0x17c2d8193f37034ce61d1307870e3e940afd5d16aa88bb70cea9a50aa47e8de0),
        Fr.wrap(0x16dbf094f8d6f1e7dbc6216836f541d7c0004fed9fa95b5f317f34596022c8a1),
        Fr.wrap(0x35b3896d2890020474239236680b8a25125f8a260dacec331c5faff0f0e4cad8),
        Fr.wrap(0xd5fcd50bb7998d61d302824dd6fa86350844f32d39605e2f7964800a758fe954),
        Fr.wrap(0x89723c608026a6f7381da210638c3a96daf740a9fb39a0830fbd3ef3a8735249),
        Fr.wrap(0xf676021b23b03999636594a8dc172d185e75637bdd17a353b17ff92aa7b3ed4f),
        Fr.wrap(0xda8e329b843a38214d7a251da24e9a0ddae4803db4a8030835b555a7c9d5c521),
        Fr.wrap(0x760d2a3f37f1bec5259e64e145999d55eeb2248ccd8fd78d392703b274b95ddd),
        Fr.wrap(0x311c4a5fc45601e4429837c103ebe66c98230afd220a9949f9df0bfcbb0336c7),
        Fr.wrap(0x79e7e345645cf06bd22ebfd7184cc8c84aed15725c362fcf1a6f9382c3312d2a),
        Fr.wrap(0xfc5cbdbca5e557fe12fbf1d72723d976b180a965b433f2f20e13093627ea0e71),
        Fr.wrap(0xf9ad55f509db0c2d5774f772fd6aee757bd37e46e5c3424d4ff6600379409e6b),
        Fr.wrap(0x0eed1610e701c5e45537359e73a4d063d4f4f5f16c229403443518832f7375c3),
        Fr.wrap(0xeb5e1c17bc9cc2ed99c5da7c8e90c80fbf6a6612f92a1b108fb07708cca381bf),
        Fr.wrap(0xb3eb7df7080cf5c1959b490e442139df6afe56c9bb7b0ea60cf3a6100546e4be),
        Fr.wrap(0x3a1a8e74a79bdd18e8afbe27057e4fa0a2c3806ace488f5258b9b74e06c29744),
        Fr.wrap(0x3ee483a3bd226709db6f7782e734518f83d5ffdb2733ad73ddc219beeb482424),
        Fr.wrap(0x7b464bcd9a795510612746c632c0376d2af67ddf15c3e2dd748e6bc862a07820),
        Fr.wrap(0x4499e78f5d2193fe96f616193f9fa78f8316c294ec07ed3f422ee27f89cdfb41),
        Fr.wrap(0x88c694dde769528b39b8351d1001c48c93294ca5a4c264f4f60e88dab5ac0c95),
        Fr.wrap(0x774a5b97284e3e535c146d4f6c93d6d37e3a76098ad19933d59b4b67611a4fbc),
        Fr.wrap(0x51bd5e9d5d3513a6214ab1452695b25118ecacfe05a81b47a7dd019ac0d30577),
        Fr.wrap(0xcbe97a8e1fc66f35acfd209e85b0bda197a4f885dbdb3d219e5f175246c2bdef),
        Fr.wrap(0x4248a2fbafc59de597c1ebe6fad80d636e75bd8c81c8d795a3db9282e05d3f9b),
        Fr.wrap(0xbf3f672fed63c9fda0b1f1dc5100fd6f9137b5712789d24a727b715f9fa06777),
        Fr.wrap(0x105085e77ae7a917a086a7a6673acf39492402f578e87b27579fddbbd590d3fc),
        Fr.wrap(0x0e37494b3deb0b2a3812652cf35ece9cb8007dbb8b4197b7e687b1f59614438a),
        Fr.wrap(0xd0b4467fe177e3e2034809f5c976e3d44bc9ab32c573bcca3537c624cc4d5bd5),
        Fr.wrap(0x35e6da09ab081e71f8150ec79eac6ff2669de0473c9a40ce8fd0cf2d2c6b6d8c),
        Fr.wrap(0x63c2def5fd4e0ac2d6a9c592b50c02c95f73f4ed3befef67db11e536ad5de30d),
        Fr.wrap(0x7e1f455dc4ec6e3daa68716a3bd209324c206f0cc37bbed921bcdfe535835e16),
        Fr.wrap(0x6019f8384eb1fbdd5d239c717241609fb24db49958eccd48de8ed626295c4841),
        Fr.wrap(0x179b19aa432721e1dd5deb1f1d04808f4eded505055c57db3471169a9a4fe78b),
        Fr.wrap(0xcc404fedd3e418a1e593473659dd9a57306d321370fe544ef7a813054351fd18),
        Fr.wrap(0xc9913a9d6eeddb885e969f15a6a9b5f6fa6500f097e06b762e1ddc361c427548),
        Fr.wrap(0xee10b8050bdc66d66839251c8d15813d9166ba0f26d28f69a88d9fc386144519),
        Fr.wrap(0xd8a37e14c73a0e4a10f03797cf82e9be78a0415b0415c391fd14414e9379fa71),
        Fr.wrap(0x28a2e8ddbed3c6256d87ef9202faec7c5dd63ca1f61d8d72f320f495597c16f4),
        Fr.wrap(0x101f45a419095bb4b84f99fd1d7ebc1367e0a126ad90b92a3a8133df6c8c06f4),
        Fr.wrap(0x4fb4a98c67ede1b55b9d3b2811d0c73488042480c7a99e5316b375fcddce1030),
        Fr.wrap(0x7bda0d4ceb10eb0aa2194b49984c52ff45a98e78e49c0db4647e466f64b47c72),
        Fr.wrap(0xb5e920313739f82748538381c1d9cc245b54ddc20160c937794480ba083ca72c),
        Fr.wrap(0xb795a898e52b3c978fb40ff58eee1315dde93c8e66cdfedd391623d032b5c1dc),
        Fr.wrap(0xde44455a4a1ecf089050764543caba4b8e3a28218467db4a0dfda1e57b5fd3f4),
        Fr.wrap(0xeeb3be3854352afe14befe89d6e46a40483eaf8a7544837776e31372215f9d4b),
        Fr.wrap(0x9f6336fb867e03d0240674d823c740909cc9406827100cf2f8be6d459e69dc47),
        Fr.wrap(0x1b4635917b217e929a6d795e3457b429974951acb2fcd26ba543b6b34100db71),
        Fr.wrap(0xba6e7c22509c5d369f414665a271556a9005b59218ad2d0eb09a7494546873c6)
    ];
        

/*
    function challengeVector(string memory label) external returns (bytes[] memory transcript){
        bytes[] memory ret;
        ret.push(challenges[0]);
        ret.push(challenges[1]);
        return ret;

    }
*/
    
    function challengeScalar(string memory label, uint256 index) external returns (Fr challenge){

        return challenges[index];
    }


}
