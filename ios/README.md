# Jolt Prover iOS App

A native iOS application that uses the Jolt zkVM prover via FFI to generate zero-knowledge proofs on iPhone and iPad devices.

## Features

- Native SwiftUI interface optimized for iOS
- File picker integration for selecting ELF and preprocessing files
- Real-time status updates during proof generation
- Proof files saved to app's Documents directory
- Support for iPhone and iPad (iOS 16+)

## Architecture

The app consists of several key components:

1. **jolt-ffi**: Rust static library providing C FFI bindings
2. **JoltFFI.swift**: Swift wrapper providing a safe, idiomatic Swift API
3. **JoltProverViewModel**: Business logic and state management
4. **ContentView**: SwiftUI user interface

## Prerequisites

Before building the iOS app, ensure you have:

- macOS with Xcode 15+ installed
- Rust toolchain installed (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- iOS target toolchains for Rust:
  ```bash
  rustup target add aarch64-apple-ios         # iPhone/iPad devices
  rustup target add aarch64-apple-ios-sim     # Apple Silicon simulator
  rustup target add x86_64-apple-ios          # Intel simulator
  ```

## Building the Project

### Step 1: Build the Rust Library

From the `ios/` directory, run the build script:

```bash
cd ios
./build_ios_lib.sh
```

This script will:
- Compile `jolt-ffi` for all iOS architectures
- Create universal libraries for simulator (arm64 + x86_64)
- Copy the compiled libraries and header files to `JoltProver/libs/`

The build process takes several minutes as it compiles the entire Jolt zkVM stack for multiple architectures.

### Step 2: Open in Xcode

```bash
open JoltProver/JoltProver.xcodeproj
```

### Step 3: Configure Code Signing

1. In Xcode, select the **JoltProver** project in the navigator
2. Select the **JoltProver** target
3. Go to **Signing & Capabilities** tab
4. Select your development team or configure automatic signing

### Step 4: Build and Run

1. Select your target device or simulator from the device menu
2. Press **⌘R** to build and run the app

## Usage

### Preparing Input Files

Before using the app, you need:

1. **ELF File**: A compiled RISC-V guest program (`.elf`)
2. **Preprocessing File**: Generated preprocessing data (`.bin`)

You can generate these files using the Jolt CLI or the `prepare-guest` binary from jolt-ffi.

#### Example: Generating Files for Fibonacci

```bash
# From the jolt repository root
cd examples/fibonacci

# Build the guest program
cargo build --release --target riscv32im-unknown-none-elf

# This generates: target/riscv32im-unknown-none-elf/release/fibonacci-guest

# Generate preprocessing (you'll need to implement this step based on your workflow)
```

### Running the App

1. **Launch the app** on your iPhone or iPad
2. **Select Preprocessing File**: Tap "Select Preprocessing File" and choose your preprocessing binary
3. **Select ELF File**: Tap "Select ELF File" and choose your compiled RISC-V guest program
4. **Generate Proof**: Tap "Generate Proof" button

The app will:
- Load the preprocessing data
- Create a prover from the ELF file
- Generate a zero-knowledge proof
- Save the proof to the app's Documents directory as `proof.bin`

### Accessing Generated Proofs

Generated proofs are saved to the app's Documents directory. You can access them via:

1. **Files app** (if you enable file sharing in Xcode)
2. **Xcode's Devices and Simulators window**:
   - Window → Devices and Simulators
   - Select your device
   - Select the JoltProver app
   - Download the Documents folder

## Project Structure

```
ios/
├── build_ios_lib.sh                 # Build script for Rust library
├── README.md                         # This file
└── JoltProver/
    ├── JoltProver.xcodeproj/        # Xcode project
    ├── libs/                         # Compiled libraries (generated)
    │   ├── libjolt_ffi_device.a     # iOS device library
    │   ├── libjolt_ffi_sim.a        # iOS simulator library
    │   ├── jolt-ffi.h               # C header file
    │   └── module.modulemap         # Swift module map
    └── JoltProver/                   # Swift source files
        ├── JoltProverApp.swift      # App entry point
        ├── ContentView.swift        # Main UI view
        ├── JoltProverViewModel.swift # Business logic
        ├── JoltFFI.swift            # Swift FFI wrapper
        └── Assets.xcassets/         # App assets
```

## API Reference

### JoltProver Class

Swift wrapper around the Jolt FFI:

```swift
// Initialize with preprocessing file
let prover = try JoltProver(preprocessingPath: "/path/to/preprocessing.bin")

// Generate prover from ELF
try prover.generateProver(
    elfPath: "/path/to/guest.elf",
    inputs: nil,
    untrustedAdvice: nil,
    trustedAdvice: nil
)

// Generate proof
try prover.prove(outputPath: "/path/to/proof.bin")
```

### JoltFFIError

Error types thrown by the FFI wrapper:

- `preprocessingLoadFailed(String)`: Failed to load preprocessing file
- `proverGenerationFailed(String)`: Failed to create prover from ELF
- `provingFailed(String)`: Failed to generate proof
- `invalidFilePath`: Invalid file path provided
- `unknownError`: Unknown error occurred

## Performance Notes

- Proof generation is computationally intensive and may take several minutes on mobile devices
- The app runs proof generation on the main thread; consider using background execution for better UX
- Memory usage can be high depending on the size of the guest program and trace length

## Troubleshooting

### Build Errors

**"library not found for -ljolt_ffi_sim"**
- Run `./build_ios_lib.sh` to generate the static libraries
- Check that `libs/` directory contains the `.a` files

**"No such module 'JoltFFIC'"**
- Verify `module.modulemap` exists in `libs/`
- Check that `SWIFT_INCLUDE_PATHS` is set to `$(PROJECT_DIR)/libs` in Xcode build settings

### Runtime Errors

**"Failed to load preprocessing"**
- Verify the preprocessing file is valid and compatible with jolt-ffi version
- Check that the file is accessible from the app's sandbox

**"Failed to generate prover from ELF"**
- Ensure the ELF file is a valid RISC-V binary compiled with the correct target
- Verify the preprocessing data matches the ELF binary

## Development

### Rebuilding the Rust Library

After making changes to `jolt-ffi`, rebuild the library:

```bash
cd ios
./build_ios_lib.sh
```

Then clean and rebuild in Xcode (**⌘⇧K** then **⌘B**).

### Adding New FFI Functions

1. Add the C function to `jolt-ffi/src/lib.rs`
2. Rebuild the library to regenerate the header
3. Add Swift wrapper to `JoltFFI.swift`
4. Update the ViewModel and UI as needed

## License

See the LICENSE file in the repository root.

## Support

For issues and questions:
- Jolt zkVM: https://github.com/a16z/jolt
- iOS App: File an issue in the main Jolt repository
