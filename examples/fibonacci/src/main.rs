use jolt_sdk::{serialize_and_print_size, TrustedAdvice};
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_dummy(target_dir);

    let prover_preprocessing = guest::preprocess_prover_dummy(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_dummy(&prover_preprocessing);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_dummy = guest::build_prover_dummy(program, prover_preprocessing.clone());
    let verify_dummy = guest::build_verifier_dummy(verifier_preprocessing);

    // 128 random 16-byte arrays
    let a1: [u8; 16] = [212, 153, 184, 212, 89, 14, 22, 91, 14, 141, 18, 182, 122, 243, 106, 53];
    let a2: [u8; 16] = [213, 82, 157, 241, 9, 174, 228, 45, 16, 1, 249, 183, 122, 112, 186, 51];
    let a3: [u8; 16] = [95, 121, 63, 64, 237, 108, 57, 231, 233, 204, 72, 239, 160, 146, 84, 114];
    let a4: [u8; 16] = [158, 98, 76, 157, 150, 157, 15, 211, 238, 14, 6, 135, 123, 210, 19, 186];
    let a5: [u8; 16] = [179, 73, 16, 197, 67, 10, 0, 191, 0, 149, 217, 144, 111, 195, 99, 83];
    let a6: [u8; 16] = [145, 180, 12, 83, 30, 248, 26, 137, 82, 115, 216, 15, 30, 18, 237, 69];
    let a7: [u8; 16] = [198, 172, 66, 164, 212, 68, 171, 122, 250, 78, 128, 182, 52, 139, 71, 126];
    let a8: [u8; 16] = [114, 137, 165, 194, 103, 118, 237, 83, 73, 150, 230, 172, 129, 98, 141, 229];
    let a9: [u8; 16] = [155, 182, 29, 71, 132, 123, 217, 26, 19, 106, 86, 235, 82, 133, 126, 192];
    let a10: [u8; 16] = [126, 40, 60, 23, 56, 160, 181, 66, 86, 28, 217, 210, 176, 130, 7, 214];
    let a11: [u8; 16] = [254, 55, 219, 107, 170, 11, 36, 65, 227, 81, 172, 25, 169, 39, 175, 249];
    let a12: [u8; 16] = [102, 212, 120, 3, 158, 36, 213, 125, 180, 61, 193, 70, 73, 109, 64, 8];
    let a13: [u8; 16] = [58, 190, 89, 151, 110, 42, 140, 226, 196, 8, 206, 129, 159, 221, 136, 224];
    let a14: [u8; 16] = [119, 28, 63, 186, 183, 37, 184, 126, 52, 143, 223, 181, 110, 39, 14, 27];
    let a15: [u8; 16] = [237, 20, 31, 196, 165, 201, 210, 154, 4, 155, 83, 68, 187, 180, 49, 35];
    let a16: [u8; 16] = [98, 57, 147, 203, 102, 63, 30, 42, 224, 58, 180, 252, 160, 252, 115, 153];
    let a17: [u8; 16] = [118, 137, 229, 212, 184, 2, 54, 51, 107, 253, 182, 14, 208, 41, 176, 90];
    let a18: [u8; 16] = [121, 92, 236, 63, 68, 37, 57, 78, 19, 242, 200, 11, 143, 253, 219, 143];
    let a19: [u8; 16] = [77, 55, 120, 212, 159, 33, 209, 244, 171, 181, 216, 152, 14, 23, 133, 191];
    let a20: [u8; 16] = [84, 86, 180, 15, 110, 5, 86, 163, 52, 93, 151, 176, 14, 236, 138, 109];
    let a21: [u8; 16] = [94, 212, 179, 147, 249, 45, 72, 182, 102, 212, 140, 233, 95, 6, 179, 212];
    let a22: [u8; 16] = [72, 226, 226, 25, 48, 199, 57, 254, 83, 158, 104, 69, 206, 81, 160, 65];
    let a23: [u8; 16] = [105, 12, 81, 242, 140, 145, 235, 16, 2, 99, 109, 187, 28, 228, 163, 49];
    let a24: [u8; 16] = [132, 85, 29, 204, 216, 196, 253, 178, 101, 179, 70, 132, 84, 74, 109, 21];
    let a25: [u8; 16] = [168, 129, 159, 66, 30, 18, 153, 152, 228, 64, 113, 130, 89, 47, 58, 195];
    let a26: [u8; 16] = [201, 230, 48, 149, 92, 217, 233, 103, 243, 211, 35, 224, 189, 249, 79, 184];
    let a27: [u8; 16] = [44, 222, 133, 179, 97, 195, 68, 132, 177, 162, 2, 159, 193, 59, 121, 68];
    let a28: [u8; 16] = [216, 50, 61, 216, 85, 197, 221, 188, 49, 142, 1, 49, 135, 132, 147, 186];
    let a29: [u8; 16] = [197, 238, 212, 68, 213, 130, 243, 94, 10, 84, 110, 213, 97, 232, 60, 237];
    let a30: [u8; 16] = [203, 26, 15, 190, 232, 76, 48, 228, 116, 45, 78, 89, 77, 33, 81, 247];
    let a31: [u8; 16] = [61, 204, 79, 19, 84, 187, 201, 12, 154, 16, 109, 203, 178, 176, 52, 231];
    let a32: [u8; 16] = [231, 180, 173, 35, 25, 34, 240, 119, 203, 21, 138, 8, 23, 2, 51, 244];
    let a33: [u8; 16] = [147, 82, 193, 58, 214, 167, 31, 89, 243, 112, 76, 201, 134, 55, 228, 19];
    let a34: [u8; 16] = [63, 241, 108, 172, 95, 186, 220, 43, 159, 67, 132, 251, 88, 203, 14, 177];
    let a35: [u8; 16] = [206, 119, 45, 163, 82, 234, 157, 71, 198, 140, 53, 225, 166, 91, 208, 36];
    let a36: [u8; 16] = [178, 54, 217, 126, 191, 68, 243, 102, 37, 189, 156, 84, 231, 147, 62, 215];
    let a37: [u8; 16] = [91, 168, 23, 246, 134, 79, 201, 58, 175, 112, 239, 46, 183, 95, 152, 127];
    let a38: [u8; 16] = [242, 105, 186, 159, 72, 213, 34, 148, 227, 66, 191, 103, 254, 38, 169, 81];
    let a39: [u8; 16] = [167, 214, 51, 138, 96, 179, 241, 24, 207, 155, 82, 193, 60, 234, 117, 146];
    let a40: [u8; 16] = [43, 189, 126, 235, 164, 71, 208, 93, 150, 37, 176, 109, 222, 56, 143, 198];
    let a41: [u8; 16] = [184, 67, 250, 101, 216, 142, 59, 178, 85, 231, 14, 163, 96, 209, 134, 47];
    let a42: [u8; 16] = [122, 195, 78, 237, 154, 41, 188, 115, 202, 56, 169, 132, 245, 63, 218, 90];
    let a43: [u8; 16] = [253, 136, 169, 42, 207, 84, 151, 226, 19, 178, 101, 244, 157, 70, 123, 186];
    let a44: [u8; 16] = [68, 215, 102, 189, 46, 133, 240, 173, 86, 219, 152, 35, 198, 111, 48, 225];
    let a45: [u8; 16] = [177, 50, 237, 124, 191, 58, 145, 212, 79, 166, 253, 96, 143, 30, 203, 117];
    let a46: [u8; 16] = [234, 147, 80, 213, 106, 39, 176, 249, 142, 75, 208, 161, 54, 227, 90, 183];
    let a47: [u8; 16] = [126, 193, 246, 139, 72, 185, 118, 51, 224, 157, 90, 243, 136, 29, 162, 215];
    let a48: [u8; 16] = [48, 161, 234, 167, 100, 33, 206, 139, 72, 245, 128, 61, 194, 87, 250, 143];
    let a49: [u8; 16] = [219, 112, 45, 178, 251, 144, 77, 210, 103, 36, 169, 62, 235, 168, 81, 14];
    let a50: [u8; 16] = [156, 89, 222, 115, 48, 181, 254, 147, 40, 213, 106, 199, 92, 25, 158, 71];
    let a51: [u8; 16] = [83, 176, 109, 242, 175, 68, 1, 134, 227, 160, 53, 186, 119, 212, 105, 38];
    let a52: [u8; 16] = [230, 123, 196, 49, 142, 235, 128, 61, 194, 87, 20, 153, 246, 139, 72, 205];
    let a53: [u8; 16] = [158, 51, 184, 117, 210, 103, 36, 169, 62, 195, 88, 221, 114, 47, 180, 73];
    let a54: [u8; 16] = [6, 139, 232, 165, 58, 191, 84, 17, 150, 243, 136, 29, 162, 95, 228, 121];
    let a55: [u8; 16] = [254, 147, 40, 173, 66, 199, 132, 25, 118, 211, 104, 237, 130, 23, 156, 89];
    let a56: [u8; 16] = [182, 75, 8, 141, 234, 127, 60, 193, 86, 19, 152, 45, 178, 71, 4, 137];
    let a57: [u8; 16] = [70, 3, 136, 229, 122, 55, 188, 81, 14, 147, 240, 133, 26, 159, 52, 185];
    let a58: [u8; 16] = [238, 171, 64, 197, 90, 223, 116, 9, 142, 235, 168, 61, 194, 127, 60, 153];
    let a59: [u8; 16] = [246, 179, 72, 205, 98, 31, 164, 57, 190, 83, 16, 149, 242, 135, 28, 161];
    let a60: [u8; 16] = [214, 107, 200, 93, 186, 79, 12, 145, 38, 171, 64, 197, 250, 183, 76, 169];
    let a61: [u8; 16] = [102, 35, 128, 221, 154, 47, 180, 113, 6, 99, 232, 165, 58, 151, 244, 137];
    let a62: [u8; 16] = [30, 163, 56, 189, 82, 215, 148, 41, 174, 67, 0, 133, 226, 119, 212, 105];
    let a63: [u8; 16] = [198, 91, 24, 157, 250, 143, 36, 169, 62, 195, 88, 181, 114, 207, 100, 33];
    let a64: [u8; 16] = [166, 59, 152, 245, 138, 71, 204, 97, 30, 123, 216, 149, 42, 175, 68, 201];
    let a65: [u8; 16] = [94, 27, 160, 253, 146, 79, 212, 105, 38, 131, 224, 157, 50, 183, 76, 9];
    let a66: [u8; 16] = [142, 235, 128, 21, 154, 87, 220, 113, 46, 179, 72, 5, 138, 231, 124, 17];
    let a67: [u8; 16] = [150, 243, 96, 189, 82, 15, 148, 241, 134, 27, 160, 53, 186, 79, 12, 145];
    let a68: [u8; 16] = [238, 131, 24, 117, 210, 103, 196, 89, 222, 155, 48, 181, 74, 7, 140, 233];
    let a69: [u8; 16] = [86, 219, 152, 45, 178, 111, 4, 97, 190, 83, 16, 149, 42, 175, 228, 121];
    let a70: [u8; 16] = [54, 187, 80, 213, 106, 39, 172, 65, 198, 91, 184, 77, 10, 143, 236, 129];
    let a71: [u8; 16] = [22, 155, 248, 141, 34, 167, 60, 193, 126, 19, 152, 245, 138, 31, 164, 97];
    let a72: [u8; 16] = [230, 123, 56, 189, 82, 215, 108, 41, 174, 67, 200, 93, 186, 119, 212, 65];
    let a73: [u8; 16] = [158, 251, 144, 37, 170, 63, 196, 129, 22, 115, 208, 101, 234, 127, 60, 153];
    let a74: [u8; 16] = [246, 139, 32, 165, 98, 231, 124, 17, 150, 43, 176, 109, 202, 95, 28, 161];
    let a75: [u8; 16] = [254, 147, 80, 13, 146, 239, 132, 25, 158, 51, 184, 117, 210, 103, 36, 169];
    let a76: [u8; 16] = [62, 195, 88, 221, 114, 47, 180, 73, 6, 139, 232, 165, 58, 191, 84, 177];
    let a77: [u8; 16] = [110, 203, 96, 29, 162, 255, 148, 41, 174, 67, 0, 133, 226, 159, 52, 185];
    let a78: [u8; 16] = [78, 11, 144, 237, 170, 63, 156, 249, 142, 35, 168, 101, 234, 127, 220, 113];
    let a79: [u8; 16] = [46, 179, 72, 205, 98, 231, 164, 57, 190, 83, 216, 149, 42, 175, 68, 1];
    let a80: [u8; 16] = [134, 227, 120, 53, 186, 119, 52, 185, 78, 171, 64, 197, 90, 23, 156, 249];
    let a81: [u8; 16] = [182, 75, 8, 141, 74, 167, 60, 193, 86, 19, 152, 245, 138, 71, 204, 137];
    let a82: [u8; 16] = [30, 123, 216, 109, 42, 175, 68, 201, 94, 227, 160, 53, 146, 239, 132, 25];
    let a83: [u8; 16] = [158, 251, 144, 77, 10, 143, 236, 169, 62, 195, 128, 21, 154, 247, 140, 33];
    let a84: [u8; 16] = [166, 99, 232, 125, 218, 151, 44, 177, 70, 3, 136, 229, 122, 55, 188, 81];
    let a85: [u8; 16] = [14, 147, 240, 173, 66, 199, 92, 25, 118, 211, 104, 37, 170, 63, 196, 89];
    let a86: [u8; 16] = [222, 115, 48, 181, 114, 7, 140, 233, 126, 59, 192, 85, 18, 151, 244, 137];
    let a87: [u8; 16] = [70, 163, 56, 189, 122, 215, 148, 41, 134, 67, 200, 93, 186, 79, 12, 145];
    let a88: [u8; 16] = [238, 171, 104, 197, 90, 23, 116, 9, 142, 75, 208, 101, 194, 87, 20, 153];
    let a89: [u8; 16] = [246, 179, 112, 45, 178, 71, 4, 137, 230, 163, 96, 29, 122, 215, 108, 241];
    let a90: [u8; 16] = [174, 107, 40, 173, 66, 199, 132, 65, 198, 131, 64, 197, 90, 183, 76, 9];
    let a91: [u8; 16] = [102, 35, 168, 61, 154, 87, 220, 113, 6, 99, 192, 125, 58, 191, 84, 17];
    let a92: [u8; 16] = [150, 43, 176, 69, 2, 135, 228, 161, 54, 187, 80, 13, 146, 239, 132, 225];
    let a93: [u8; 16] = [118, 51, 184, 77, 10, 143, 36, 169, 102, 195, 88, 221, 114, 47, 140, 233];
    let a94: [u8; 16] = [166, 59, 192, 85, 218, 111, 44, 177, 70, 203, 96, 189, 82, 15, 148, 41];
    let a95: [u8; 16] = [174, 67, 0, 133, 226, 119, 212, 105, 38, 171, 64, 157, 250, 143, 76, 9];
    let a96: [u8; 16] = [142, 235, 128, 61, 194, 87, 20, 153, 46, 139, 232, 165, 98, 31, 124, 217];
    let a97: [u8; 16] = [150, 83, 16, 149, 242, 175, 108, 41, 134, 227, 160, 93, 186, 119, 52, 185];
    let a98: [u8; 16] = [78, 11, 144, 237, 130, 23, 156, 89, 222, 155, 88, 21, 154, 247, 140, 73];
    let a99: [u8; 16] = [6, 139, 72, 205, 138, 71, 4, 97, 190, 123, 216, 109, 202, 95, 28, 161];
    let a100: [u8; 16] = [254, 187, 80, 213, 146, 79, 172, 65, 158, 91, 24, 157, 50, 183, 116, 9];
    let a101: [u8; 16] = [182, 75, 168, 101, 34, 127, 60, 193, 86, 219, 152, 85, 178, 111, 204, 137];
    let a102: [u8; 16] = [70, 3, 136, 69, 162, 255, 188, 121, 14, 147, 80, 213, 106, 39, 132, 25];
    let a103: [u8; 16] = [118, 211, 144, 77, 170, 103, 196, 129, 62, 195, 128, 61, 154, 87, 180, 113];
    let a104: [u8; 16] = [6, 99, 232, 165, 58, 151, 84, 217, 110, 43, 176, 109, 42, 175, 68, 1];
    let a105: [u8; 16] = [134, 67, 200, 133, 26, 159, 92, 25, 158, 251, 184, 117, 50, 183, 76, 169];
    let a106: [u8; 16] = [102, 195, 88, 181, 74, 7, 140, 33, 166, 99, 32, 165, 218, 151, 244, 137];
    let a107: [u8; 16] = [30, 163, 96, 29, 122, 55, 148, 241, 174, 107, 200, 93, 26, 119, 212, 105];
    let a108: [u8; 16] = [198, 131, 64, 157, 90, 223, 156, 249, 182, 115, 48, 181, 114, 47, 140, 73];
    let a109: [u8; 16] = [246, 139, 232, 125, 98, 31, 164, 97, 190, 83, 216, 149, 202, 135, 28, 161];
    let a110: [u8; 16] = [94, 187, 120, 53, 146, 79, 12, 105, 38, 171, 104, 237, 90, 23, 116, 9];
    let a111: [u8; 16] = [142, 235, 168, 61, 34, 127, 220, 153, 46, 179, 112, 45, 18, 151, 84, 177];
    let a112: [u8; 16] = [110, 43, 136, 229, 162, 95, 28, 121, 54, 147, 80, 13, 106, 199, 132, 65];
    let a113: [u8; 16] = [158, 91, 184, 77, 170, 63, 36, 129, 22, 155, 88, 221, 154, 247, 180, 73];
    let a114: [u8; 16] = [166, 59, 192, 125, 58, 191, 44, 217, 150, 83, 176, 69, 42, 135, 68, 201];
    let a115: [u8; 16] = [134, 227, 160, 93, 26, 159, 212, 145, 78, 11, 64, 157, 250, 183, 116, 249];
    let a116: [u8; 16] = [22, 115, 208, 141, 74, 167, 100, 33, 126, 19, 152, 85, 178, 71, 204, 97];
    let a117: [u8; 16] = [30, 163, 96, 189, 82, 55, 148, 81, 174, 67, 200, 133, 186, 79, 132, 25];
    let a118: [u8; 16] = [118, 251, 144, 37, 10, 103, 196, 89, 62, 155, 88, 181, 154, 7, 140, 233];
    let a119: [u8; 16] = [166, 99, 32, 125, 218, 111, 244, 177, 110, 203, 176, 69, 122, 215, 68, 1];
    let a120: [u8; 16] = [94, 27, 120, 213, 146, 39, 92, 185, 38, 131, 64, 197, 50, 143, 236, 169];
    let a121: [u8; 16] = [62, 235, 168, 101, 194, 87, 60, 193, 86, 179, 72, 205, 18, 111, 44, 137];
    let a122: [u8; 16] = [230, 123, 56, 29, 2, 95, 188, 121, 134, 227, 80, 173, 186, 119, 132, 225];
    let a123: [u8; 16] = [158, 51, 24, 117, 170, 143, 196, 89, 222, 115, 208, 141, 114, 167, 100, 33];
    let a124: [u8; 16] = [6, 139, 72, 165, 138, 31, 84, 17, 70, 3, 176, 109, 162, 55, 148, 241];
    let a125: [u8; 16] = [254, 147, 200, 93, 106, 199, 92, 25, 78, 171, 24, 157, 90, 183, 76, 169];
    let a126: [u8; 16] = [102, 35, 128, 221, 74, 47, 180, 113, 126, 59, 152, 245, 178, 71, 44, 177];
    let a127: [u8; 16] = [70, 163, 136, 29, 162, 95, 188, 81, 174, 107, 200, 133, 26, 119, 52, 145];
    let a128: [u8; 16] = [238, 131, 104, 237, 250, 143, 156, 249, 22, 115, 48, 181, 154, 247, 140, 233];

    let (trusted_advice_commitment, _hint) = guest::commit_trusted_advice_dummy(
        TrustedAdvice::new(a1),
        TrustedAdvice::new(a2),
        TrustedAdvice::new(a3),
        TrustedAdvice::new(a4),
        TrustedAdvice::new(a5),
        TrustedAdvice::new(a6),
        TrustedAdvice::new(a7),
        TrustedAdvice::new(a8),
        TrustedAdvice::new(a9),
        TrustedAdvice::new(a10),
        TrustedAdvice::new(a11),
        TrustedAdvice::new(a12),
        TrustedAdvice::new(a13),
        TrustedAdvice::new(a14),
        TrustedAdvice::new(a15),
        TrustedAdvice::new(a16),
        TrustedAdvice::new(a17),
        TrustedAdvice::new(a18),
        TrustedAdvice::new(a19),
        TrustedAdvice::new(a20),
        TrustedAdvice::new(a21),
        TrustedAdvice::new(a22),
        TrustedAdvice::new(a23),
        TrustedAdvice::new(a24),
        TrustedAdvice::new(a25),
        TrustedAdvice::new(a26),
        TrustedAdvice::new(a27),
        TrustedAdvice::new(a28),
        TrustedAdvice::new(a29),
        TrustedAdvice::new(a30),
        TrustedAdvice::new(a31),
        TrustedAdvice::new(a32),
        TrustedAdvice::new(a33),
        TrustedAdvice::new(a34),
        TrustedAdvice::new(a35),
        TrustedAdvice::new(a36),
        TrustedAdvice::new(a37),
        TrustedAdvice::new(a38),
        TrustedAdvice::new(a39),
        TrustedAdvice::new(a40),
        TrustedAdvice::new(a41),
        TrustedAdvice::new(a42),
        TrustedAdvice::new(a43),
        TrustedAdvice::new(a44),
        TrustedAdvice::new(a45),
        TrustedAdvice::new(a46),
        TrustedAdvice::new(a47),
        TrustedAdvice::new(a48),
        TrustedAdvice::new(a49),
        TrustedAdvice::new(a50),
        TrustedAdvice::new(a51),
        TrustedAdvice::new(a52),
        TrustedAdvice::new(a53),
        TrustedAdvice::new(a54),
        TrustedAdvice::new(a55),
        TrustedAdvice::new(a56),
        TrustedAdvice::new(a57),
        TrustedAdvice::new(a58),
        TrustedAdvice::new(a59),
        TrustedAdvice::new(a60),
        TrustedAdvice::new(a61),
        TrustedAdvice::new(a62),
        TrustedAdvice::new(a63),
        TrustedAdvice::new(a64),
        TrustedAdvice::new(a65),
        TrustedAdvice::new(a66),
        TrustedAdvice::new(a67),
        TrustedAdvice::new(a68),
        TrustedAdvice::new(a69),
        TrustedAdvice::new(a70),
        TrustedAdvice::new(a71),
        TrustedAdvice::new(a72),
        TrustedAdvice::new(a73),
        TrustedAdvice::new(a74),
        TrustedAdvice::new(a75),
        TrustedAdvice::new(a76),
        TrustedAdvice::new(a77),
        TrustedAdvice::new(a78),
        TrustedAdvice::new(a79),
        TrustedAdvice::new(a80),
        TrustedAdvice::new(a81),
        TrustedAdvice::new(a82),
        TrustedAdvice::new(a83),
        TrustedAdvice::new(a84),
        TrustedAdvice::new(a85),
        TrustedAdvice::new(a86),
        TrustedAdvice::new(a87),
        TrustedAdvice::new(a88),
        TrustedAdvice::new(a89),
        TrustedAdvice::new(a90),
        TrustedAdvice::new(a91),
        TrustedAdvice::new(a92),
        TrustedAdvice::new(a93),
        TrustedAdvice::new(a94),
        TrustedAdvice::new(a95),
        TrustedAdvice::new(a96),
        TrustedAdvice::new(a97),
        TrustedAdvice::new(a98),
        TrustedAdvice::new(a99),
        TrustedAdvice::new(a100),
        TrustedAdvice::new(a101),
        TrustedAdvice::new(a102),
        TrustedAdvice::new(a103),
        TrustedAdvice::new(a104),
        TrustedAdvice::new(a105),
        TrustedAdvice::new(a106),
        TrustedAdvice::new(a107),
        TrustedAdvice::new(a108),
        TrustedAdvice::new(a109),
        TrustedAdvice::new(a110),
        TrustedAdvice::new(a111),
        TrustedAdvice::new(a112),
        TrustedAdvice::new(a113),
        TrustedAdvice::new(a114),
        TrustedAdvice::new(a115),
        TrustedAdvice::new(a116),
        TrustedAdvice::new(a117),
        TrustedAdvice::new(a118),
        TrustedAdvice::new(a119),
        TrustedAdvice::new(a120),
        TrustedAdvice::new(a121),
        TrustedAdvice::new(a122),
        TrustedAdvice::new(a123),
        TrustedAdvice::new(a124),
        TrustedAdvice::new(a125),
        TrustedAdvice::new(a126),
        TrustedAdvice::new(a127),
        TrustedAdvice::new(a128),
        &prover_preprocessing,
    );

    // let program_summary = guest::analyze_dummy(10);
    // program_summary
    //     .write_to_file("dummy_10.txt".into())
    //     .expect("should write");

    let trace_file = "/tmp/dummy_trace.bin";
    guest::trace_dummy_to_file(
        trace_file,
        10,
        TrustedAdvice::new(a1),
        TrustedAdvice::new(a2),
        TrustedAdvice::new(a3),
        TrustedAdvice::new(a4),
        TrustedAdvice::new(a5),
        TrustedAdvice::new(a6),
        TrustedAdvice::new(a7),
        TrustedAdvice::new(a8),
        TrustedAdvice::new(a9),
        TrustedAdvice::new(a10),
        TrustedAdvice::new(a11),
        TrustedAdvice::new(a12),
        TrustedAdvice::new(a13),
        TrustedAdvice::new(a14),
        TrustedAdvice::new(a15),
        TrustedAdvice::new(a16),
        TrustedAdvice::new(a17),
        TrustedAdvice::new(a18),
        TrustedAdvice::new(a19),
        TrustedAdvice::new(a20),
        TrustedAdvice::new(a21),
        TrustedAdvice::new(a22),
        TrustedAdvice::new(a23),
        TrustedAdvice::new(a24),
        TrustedAdvice::new(a25),
        TrustedAdvice::new(a26),
        TrustedAdvice::new(a27),
        TrustedAdvice::new(a28),
        TrustedAdvice::new(a29),
        TrustedAdvice::new(a30),
        TrustedAdvice::new(a31),
        TrustedAdvice::new(a32),
        TrustedAdvice::new(a33),
        TrustedAdvice::new(a34),
        TrustedAdvice::new(a35),
        TrustedAdvice::new(a36),
        TrustedAdvice::new(a37),
        TrustedAdvice::new(a38),
        TrustedAdvice::new(a39),
        TrustedAdvice::new(a40),
        TrustedAdvice::new(a41),
        TrustedAdvice::new(a42),
        TrustedAdvice::new(a43),
        TrustedAdvice::new(a44),
        TrustedAdvice::new(a45),
        TrustedAdvice::new(a46),
        TrustedAdvice::new(a47),
        TrustedAdvice::new(a48),
        TrustedAdvice::new(a49),
        TrustedAdvice::new(a50),
        TrustedAdvice::new(a51),
        TrustedAdvice::new(a52),
        TrustedAdvice::new(a53),
        TrustedAdvice::new(a54),
        TrustedAdvice::new(a55),
        TrustedAdvice::new(a56),
        TrustedAdvice::new(a57),
        TrustedAdvice::new(a58),
        TrustedAdvice::new(a59),
        TrustedAdvice::new(a60),
        TrustedAdvice::new(a61),
        TrustedAdvice::new(a62),
        TrustedAdvice::new(a63),
        TrustedAdvice::new(a64),
        TrustedAdvice::new(a65),
        TrustedAdvice::new(a66),
        TrustedAdvice::new(a67),
        TrustedAdvice::new(a68),
        TrustedAdvice::new(a69),
        TrustedAdvice::new(a70),
        TrustedAdvice::new(a71),
        TrustedAdvice::new(a72),
        TrustedAdvice::new(a73),
        TrustedAdvice::new(a74),
        TrustedAdvice::new(a75),
        TrustedAdvice::new(a76),
        TrustedAdvice::new(a77),
        TrustedAdvice::new(a78),
        TrustedAdvice::new(a79),
        TrustedAdvice::new(a80),
        TrustedAdvice::new(a81),
        TrustedAdvice::new(a82),
        TrustedAdvice::new(a83),
        TrustedAdvice::new(a84),
        TrustedAdvice::new(a85),
        TrustedAdvice::new(a86),
        TrustedAdvice::new(a87),
        TrustedAdvice::new(a88),
        TrustedAdvice::new(a89),
        TrustedAdvice::new(a90),
        TrustedAdvice::new(a91),
        TrustedAdvice::new(a92),
        TrustedAdvice::new(a93),
        TrustedAdvice::new(a94),
        TrustedAdvice::new(a95),
        TrustedAdvice::new(a96),
        TrustedAdvice::new(a97),
        TrustedAdvice::new(a98),
        TrustedAdvice::new(a99),
        TrustedAdvice::new(a100),
        TrustedAdvice::new(a101),
        TrustedAdvice::new(a102),
        TrustedAdvice::new(a103),
        TrustedAdvice::new(a104),
        TrustedAdvice::new(a105),
        TrustedAdvice::new(a106),
        TrustedAdvice::new(a107),
        TrustedAdvice::new(a108),
        TrustedAdvice::new(a109),
        TrustedAdvice::new(a110),
        TrustedAdvice::new(a111),
        TrustedAdvice::new(a112),
        TrustedAdvice::new(a113),
        TrustedAdvice::new(a114),
        TrustedAdvice::new(a115),
        TrustedAdvice::new(a116),
        TrustedAdvice::new(a117),
        TrustedAdvice::new(a118),
        TrustedAdvice::new(a119),
        TrustedAdvice::new(a120),
        TrustedAdvice::new(a121),
        TrustedAdvice::new(a122),
        TrustedAdvice::new(a123),
        TrustedAdvice::new(a124),
        TrustedAdvice::new(a125),
        TrustedAdvice::new(a126),
        TrustedAdvice::new(a127),
        TrustedAdvice::new(a128),
    );
    info!("Trace file written to: {trace_file}.");

    let now = Instant::now();
    let (output, proof, io_device) = prove_dummy(
        10,
        TrustedAdvice::new(a1),
        TrustedAdvice::new(a2),
        TrustedAdvice::new(a3),
        TrustedAdvice::new(a4),
        TrustedAdvice::new(a5),
        TrustedAdvice::new(a6),
        TrustedAdvice::new(a7),
        TrustedAdvice::new(a8),
        TrustedAdvice::new(a9),
        TrustedAdvice::new(a10),
        TrustedAdvice::new(a11),
        TrustedAdvice::new(a12),
        TrustedAdvice::new(a13),
        TrustedAdvice::new(a14),
        TrustedAdvice::new(a15),
        TrustedAdvice::new(a16),
        TrustedAdvice::new(a17),
        TrustedAdvice::new(a18),
        TrustedAdvice::new(a19),
        TrustedAdvice::new(a20),
        TrustedAdvice::new(a21),
        TrustedAdvice::new(a22),
        TrustedAdvice::new(a23),
        TrustedAdvice::new(a24),
        TrustedAdvice::new(a25),
        TrustedAdvice::new(a26),
        TrustedAdvice::new(a27),
        TrustedAdvice::new(a28),
        TrustedAdvice::new(a29),
        TrustedAdvice::new(a30),
        TrustedAdvice::new(a31),
        TrustedAdvice::new(a32),
        TrustedAdvice::new(a33),
        TrustedAdvice::new(a34),
        TrustedAdvice::new(a35),
        TrustedAdvice::new(a36),
        TrustedAdvice::new(a37),
        TrustedAdvice::new(a38),
        TrustedAdvice::new(a39),
        TrustedAdvice::new(a40),
        TrustedAdvice::new(a41),
        TrustedAdvice::new(a42),
        TrustedAdvice::new(a43),
        TrustedAdvice::new(a44),
        TrustedAdvice::new(a45),
        TrustedAdvice::new(a46),
        TrustedAdvice::new(a47),
        TrustedAdvice::new(a48),
        TrustedAdvice::new(a49),
        TrustedAdvice::new(a50),
        TrustedAdvice::new(a51),
        TrustedAdvice::new(a52),
        TrustedAdvice::new(a53),
        TrustedAdvice::new(a54),
        TrustedAdvice::new(a55),
        TrustedAdvice::new(a56),
        TrustedAdvice::new(a57),
        TrustedAdvice::new(a58),
        TrustedAdvice::new(a59),
        TrustedAdvice::new(a60),
        TrustedAdvice::new(a61),
        TrustedAdvice::new(a62),
        TrustedAdvice::new(a63),
        TrustedAdvice::new(a64),
        TrustedAdvice::new(a65),
        TrustedAdvice::new(a66),
        TrustedAdvice::new(a67),
        TrustedAdvice::new(a68),
        TrustedAdvice::new(a69),
        TrustedAdvice::new(a70),
        TrustedAdvice::new(a71),
        TrustedAdvice::new(a72),
        TrustedAdvice::new(a73),
        TrustedAdvice::new(a74),
        TrustedAdvice::new(a75),
        TrustedAdvice::new(a76),
        TrustedAdvice::new(a77),
        TrustedAdvice::new(a78),
        TrustedAdvice::new(a79),
        TrustedAdvice::new(a80),
        TrustedAdvice::new(a81),
        TrustedAdvice::new(a82),
        TrustedAdvice::new(a83),
        TrustedAdvice::new(a84),
        TrustedAdvice::new(a85),
        TrustedAdvice::new(a86),
        TrustedAdvice::new(a87),
        TrustedAdvice::new(a88),
        TrustedAdvice::new(a89),
        TrustedAdvice::new(a90),
        TrustedAdvice::new(a91),
        TrustedAdvice::new(a92),
        TrustedAdvice::new(a93),
        TrustedAdvice::new(a94),
        TrustedAdvice::new(a95),
        TrustedAdvice::new(a96),
        TrustedAdvice::new(a97),
        TrustedAdvice::new(a98),
        TrustedAdvice::new(a99),
        TrustedAdvice::new(a100),
        TrustedAdvice::new(a101),
        TrustedAdvice::new(a102),
        TrustedAdvice::new(a103),
        TrustedAdvice::new(a104),
        TrustedAdvice::new(a105),
        TrustedAdvice::new(a106),
        TrustedAdvice::new(a107),
        TrustedAdvice::new(a108),
        TrustedAdvice::new(a109),
        TrustedAdvice::new(a110),
        TrustedAdvice::new(a111),
        TrustedAdvice::new(a112),
        TrustedAdvice::new(a113),
        TrustedAdvice::new(a114),
        TrustedAdvice::new(a115),
        TrustedAdvice::new(a116),
        TrustedAdvice::new(a117),
        TrustedAdvice::new(a118),
        TrustedAdvice::new(a119),
        TrustedAdvice::new(a120),
        TrustedAdvice::new(a121),
        TrustedAdvice::new(a122),
        TrustedAdvice::new(a123),
        TrustedAdvice::new(a124),
        TrustedAdvice::new(a125),
        TrustedAdvice::new(a126),
        TrustedAdvice::new(a127),
        TrustedAdvice::new(a128),
        trusted_advice_commitment,
    );
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/dummy_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/dummy_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    let is_valid = verify_dummy(10, output, io_device.panic, trusted_advice_commitment, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}
