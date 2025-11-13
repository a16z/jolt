import Foundation
import JoltFFIC

/// Swift wrapper for Jolt FFI providing a safe, idiomatic Swift API
enum JoltFFIError: Error {
    case preprocessingLoadFailed(String)
    case proverGenerationFailed(String)
    case provingFailed(String)
    case invalidFilePath
    case unknownError

    var localizedDescription: String {
        switch self {
        case .preprocessingLoadFailed(let msg):
            return "Failed to load preprocessing: \(msg)"
        case .proverGenerationFailed(let msg):
            return "Failed to generate prover: \(msg)"
        case .provingFailed(let msg):
            return "Failed to generate proof: \(msg)"
        case .invalidFilePath:
            return "Invalid file path"
        case .unknownError:
            return "Unknown error occurred"
        }
    }
}

/// Swift wrapper for JoltCpuProver
class JoltProver {
    private let preprocessing: OpaquePointer
    private var prover: OpaquePointer?

    /// Initialize by loading preprocessing from file
    init(preprocessingPath: String) throws {
        guard let cPath = preprocessingPath.cString(using: .utf8) else {
            throw JoltFFIError.invalidFilePath
        }

        guard let preprocessingPtr = jolt_prover_preprocessing_load(cPath) else {
            let error = Self.getLastError()
            throw JoltFFIError.preprocessingLoadFailed(error)
        }

        self.preprocessing = preprocessingPtr
    }

    deinit {
        // Free preprocessing
        jolt_prover_preprocessing_free(preprocessing)

        // Free prover if not already consumed by prove()
        if let prover = prover {
            jolt_cpu_prover_free(prover)
        }
    }

    /// Generate a prover from ELF file
    /// - Parameters:
    ///   - elfPath: Path to the ELF file
    ///   - inputs: Optional input bytes
    ///   - untrustedAdvice: Optional untrusted advice bytes
    ///   - trustedAdvice: Optional trusted advice bytes
    func generateProver(
        elfPath: String,
        inputs: Data? = nil,
        untrustedAdvice: Data? = nil,
        trustedAdvice: Data? = nil
    ) throws {
        // Load ELF file
        guard let elfData = try? Data(contentsOf: URL(fileURLWithPath: elfPath)) else {
            throw JoltFFIError.invalidFilePath
        }

        // Prepare input arrays
        let elfBytes = [UInt8](elfData)
        let inputBytes = inputs.map { [UInt8]($0) } ?? []
        let untrustedBytes = untrustedAdvice.map { [UInt8]($0) } ?? []
        let trustedBytes = trustedAdvice.map { [UInt8]($0) } ?? []

        // Call FFI function
        let proverPtr = elfBytes.withUnsafeBufferPointer { elfPtr in
            inputBytes.withUnsafeBufferPointer { inputPtr in
                untrustedBytes.withUnsafeBufferPointer { untrustedPtr in
                    trustedBytes.withUnsafeBufferPointer { trustedPtr in
                        jolt_cpu_prover_gen_from_elf(
                            preprocessing,
                            elfPtr.baseAddress,
                            elfPtr.count,
                            inputBytes.isEmpty ? nil : inputPtr.baseAddress,
                            inputPtr.count,
                            untrustedBytes.isEmpty ? nil : untrustedPtr.baseAddress,
                            untrustedPtr.count,
                            trustedBytes.isEmpty ? nil : trustedPtr.baseAddress,
                            trustedPtr.count
                        )
                    }
                }
            }
        }

        guard let proverPtr = proverPtr else {
            let error = Self.getLastError()
            throw JoltFFIError.proverGenerationFailed(error)
        }

        self.prover = proverPtr
    }

    /// Generate proof and save to file
    /// - Parameter outputPath: Path where the proof will be saved
    func prove(outputPath: String) throws {
        guard let prover = prover else {
            throw JoltFFIError.provingFailed("Prover not initialized. Call generateProver first.")
        }

        guard let cPath = outputPath.cString(using: .utf8) else {
            throw JoltFFIError.invalidFilePath
        }

        let result = jolt_cpu_prover_prove(prover, cPath)

        // The prover is consumed by jolt_cpu_prover_prove
        self.prover = nil

        if result != 0 {
            let error = Self.getLastError()
            throw JoltFFIError.provingFailed(error)
        }
    }

    /// Get the last error message from the FFI layer
    private static func getLastError() -> String {
        if let errorPtr = jolt_last_error() {
            return String(cString: errorPtr)
        }
        return "Unknown error"
    }
}
