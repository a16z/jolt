// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {ITranscript} from "../interfaces/ITranscript.sol";
import {Fr} from "./Fr.sol";

contract JoltTranscript is ITranscript {
    //TODO: remove
    // mocking fiat shamir values
    Fr[68] public challenges = [
        Fr.wrap(0x1f6c8fd2c091d2160c5bf6c12cff019759cdd861f99f6e045909cb1dfaead156),
        Fr.wrap(0x1efacde73c1dd3b028a1b10d17428978701f9bd8526fce74a5230ecfe3a23dc7),
        Fr.wrap(0x0ac8d152ebce541f531bd9761d5f6cd11498163f705fe20baa04bcdd1903a688),
        Fr.wrap(0x01a98937a22c04eded1e36efe665ab9415ddc8c92ac66d06f9026e07dcb98671),
        Fr.wrap(0x07b2fc9f30959eee22b0603c767cb6b63c0cdf7ef6e1e4776d4cb41c326c818a),
        Fr.wrap(0x0732f60e97ae3196edb0bd77424f834539553c9a88332f498f412f697a82f8ef),
        Fr.wrap(0x268b8aee0050647836a84c7e27fdb0c74c3a3bddbba698045b32d3d02023fd31),
        Fr.wrap(0x01b7555cfb22dbf4f82e6a1884829d2a6ed71139811e52b7542f41e47c50ba6f),
        Fr.wrap(0x1a2a384b9e112025a62be1baf8618ef662d94b6a64db54ede69859d1dbe95bd2),
        Fr.wrap(0x1639b4e77455f22d1cbc82722c1866866da975161766d99ea63b4fb3ebcb1407),
        Fr.wrap(0x2deb9d14b61f3786a287c949b78df1a9787f878b1191af2712f0b7ad55ad96b1),
        Fr.wrap(0x26d5b0d5a5a0db956b3fe04eba35d6129cf4074ad78a858bc18ea322f90dcab2),
        Fr.wrap(0x1dc3811dc396a6f11d1f32775d3ca2ef28ac7a5c98638c8d7f98ffc84c197770),
        Fr.wrap(0x0b729ed60a094d3f1d9b6a5e34e9aa7ada49e084bc7171c68e1e8c28c16f8c37),
        Fr.wrap(0x1f94e4d14918e07eeee6eea47aadec436d15ed87a4c240aadfa8f5e4e6c0869a),
        Fr.wrap(0x2716ecba39748a22e7742c5adc091dc4c45a76e623847a8e1f44eae12b1b3ced),
        Fr.wrap(0x242293efd4a1908320ea11ee6a945195e09b43ca82e14dd1cd2d156586badca1),
        Fr.wrap(0x1b35901001e6c31d671a69c44edcbbf124a90500bb0895b62df92671dc213edf),
        Fr.wrap(0x1e8a48c9d80ee1f965ada8f57937619f07b36e5b3ac8c4190fc21ec40b822204),
        Fr.wrap(0x07579fd68eb81ff6494d674b186e7cf75bb4424283454b6aede2ddac816419fd),
        Fr.wrap(0x08579ca89ef118b8d66d2c9c6b85ac87186b29e39f2cfac2bd0276bda8816232),
        Fr.wrap(0x261d9d4157eb70acdedd26ae83a9a862e65affb47d4a2fadf6ead7de3a0e94c7),
        Fr.wrap(0x284559bfca7ce8331b12ef110bd11198bd2a149653dbe559797ad0b1519bf671),
        Fr.wrap(0x2f9f48659479f3fd1201d3e82cb786f4c4e6f1b65dde045c15215c4425f66f4c),
        Fr.wrap(0x24675a06e572f522a880af548376349f2a1a16c1b141fc9d6d40860c330af775),
        Fr.wrap(0x1cd0d3964d2c1d8f1e57a23bc507257cbbb426f85e943fea0e37ea3321d3142d),
        Fr.wrap(0x0b770ceaa4ba9184e016f18ffe7f55f740db5a4e83830dc1c3569318e459a480),
        Fr.wrap(0x1a4751b9658512de577c73fc79ce3897e9d0331b2b45379b85ed402f068626af),
        Fr.wrap(0x1d34f27d08e2e1a1d149ee456e0d18aa5bb7872b0e2c27afd0b341289b6806a0),
        Fr.wrap(0x106cf46b9cc116bd81431946fecc7a386874d1712daf430f713cb2f7ac695cc5),
        Fr.wrap(0x16c33d4b6b93197a616090cf9408f18fbad40adf7ada39f9ed6395ad2dd1354e),
        Fr.wrap(0x2439d95acb3de2177033029671697d9d7f827c6991ab9a91ad29ccc94b29918d),
        Fr.wrap(0x1d813862f2546f6ab53ade1548ab79fc4bf7d724cc5f72445712229eab0e0fac),
        Fr.wrap(0x05083e14bc18698db43eaeca9af1efc67d39380ceac2e70fd1334877bae73366),
        Fr.wrap(0x2dd6c8b5c1d642e2a4960aee742a879721bd448b5839082f6fb150fa8a7114f1),
        Fr.wrap(0x2310f61c33f9714ba9069246758cdabe55e371699a2339edccf15845645ad1fe),
        Fr.wrap(0x23bd40a7ae1803cd89d8eab70d5d55fde9e85dfcde747c7f399a91ca4a753dbf),
        Fr.wrap(0x24721ae1b1238cd93fac973c54a478d9faa22ddd244eace2006657ca3319c8b9),
        Fr.wrap(0x0850b04ee3f578563163563683cacaf7c63b0dbfc7a3446add4ff0b03e687adf),
        Fr.wrap(0x0cce5b572fc4bed21ec2522cc2114f3d63695d5971256be7ad2137a31d859b9b),
        Fr.wrap(0x1fbe01c9fc2e7e305bd223a60113b564dafe70ebbb80edcfcbab56eb1de364d6),
        Fr.wrap(0x0c82edca857acb980319a8e58b758c56bd08551d3880545719358977c38ae1b4),
        Fr.wrap(0x20fd55f0812df877b5a8d499ef3aace12a589ca9c922278efb0714100b1d9055),
        Fr.wrap(0x232af7facca5c9d2be443221ad48bc13cf153c2560f7bb93f42a9a81d4d13990),
        Fr.wrap(0x00153338eb116f0d261e034d6dd09f46eac8a66e01f285a4967b4600981d2e83),
        Fr.wrap(0x2ec18ba7f854072e43b86eaa87cef520365ab85d366dcc310c360b47e6b1670d),
        Fr.wrap(0x170c8dd41474a834c850f857558dba394a27954d4a717f971d69dae34c0b3b29),
        Fr.wrap(0x095dc362164c9b6b6a3c5ffd64f9f321b660ef5e2131925bf7f30afd3c534cb2),
        Fr.wrap(0x0d03cdddabcd5ea77ae1808a5e36e967e68782362a88b3d4d8c6ee21a65a1a51),
        Fr.wrap(0x2f3aba270c63106879656a4160410d7dc31998a63d3c3abe57fe67f340a66999),
        Fr.wrap(0x2676b23e07cee11440800f06515220494ad22156e341394542fe92ea1ac6507a),
        Fr.wrap(0x1d7bd8dc5325a06b2716cd64783faece2cd0c32776213039cd53f719abbb6836),
        Fr.wrap(0x0adf435a99065949f5b677f19111d3518fae7a2fe374e2f1564f507f6ed4261f),
        Fr.wrap(0x00cbe18a5fd17a2edf2cb796e7578e8a74dc6f46d0f0baaf83637d80e73b92b8),
        Fr.wrap(0x0679c2f89d976ae1963921dbf55138786a079164a061fc86841abfc5ff523481),
        Fr.wrap(0x09213df1fedea158ba0c0b5218d101e318d0e038d63f0139ff359959bf77fa15),
        Fr.wrap(0x14227b58c23d30b32025c5e8e994dd7ae52ba2b8804bddf93794a8e769b13b78),
        Fr.wrap(0x09c7009a609a82697c0947712ac2ee6bb5f3bc06cd7008ac4223eaa5c5800e81),
        Fr.wrap(0x13c42c162d14b0a284b90dca5329e23e74f09f824ca66178ca26a7eb5877a423),
        Fr.wrap(0x24f2484cc21fcb5a83e197a0b514612dfa87d77b8e4dc9afb25feb7f9cf8ed4f),
        Fr.wrap(0x297f7397cd596162487487900a291227d5fa6508516ba061489ac6db05c3ddad),
        Fr.wrap(0x13f35c3053aff5aa572d628a760fedb03d67d82f3a042ecfb82771288cd8e155),
        Fr.wrap(0x305b3a49e9e73aae51784d2a655faaa087a8eadf8906aa6175f15d72c8a12b68),
        Fr.wrap(0x18a0fabfa196222ac714cd7daa1244b26cb06aa12d679209ef1c81037e72703a),
        Fr.wrap(0x26d705455da628e4c8edd2f097e9d153be831b13d48c79a12b6562b6424a46e9),
        Fr.wrap(0x1a7f62aad74cd91bf1524bafddce4ef82e4d22fc4f96decf0a1ef9fec7504bae),
        Fr.wrap(0x1d7f1da958ba1b98def30f992047fe8371cb2879d7533db4a3a221746ae08153),
        Fr.wrap(0x24b1edf5822e0f9e928470ac0a3d26907aa6e9505e71b0b89342aac127cf7ba4)
    ];
    /*
    function challengeVector(string memory label) external returns (bytes[] memory transcript){
        bytes[] memory ret;
        ret.push(challenges[0]);
        ret.push(challenges[1]);
        return ret;

    }
    */

    function challengeScalar(string memory, uint256 index) external view returns (Fr challenge) {
        return challenges[index];
    }
}
