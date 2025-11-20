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

var logging_init = false
func init_logging() {
    if !logging_init {
        logging_init = true
        jolt_ios_logging_init()
    }
}

/// Reset Dory global state before generating a new proof
/// This prevents issues when switching between proofs with different dimensions
func reset_dory() {
    jolt_reset_dory_globals()
}

/// Swift wrapper for JoltCpuProver
class JoltProver {
    private let preprocessing: OpaquePointer
    private var prover: OpaquePointer?

    /// Initialize by loading preprocessing from file
    init(preprocessingPath: String) throws {
        init_logging()
        
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
                            UInt(elfPtr.count),
                            inputBytes.isEmpty ? nil : inputPtr.baseAddress,
                            UInt(inputPtr.count),
                            untrustedBytes.isEmpty ? nil : untrustedPtr.baseAddress,
                            UInt(untrustedPtr.count),
                            trustedBytes.isEmpty ? nil : trustedPtr.baseAddress,
                            UInt(trustedPtr.count)
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
    static func getLastError() -> String {
        if let errorPtr = jolt_last_error() {
            return String(cString: errorPtr)
        }
        return "Unknown error"
    }
}

// MARK: - Postcard Serialization Helpers

/// Helper class for postcard serialization via FFI
class PostcardSerializer {
    /// Serialize a single UInt64 value using postcard
    static func serialize(_ value: UInt64) throws -> Data {
        guard let handle = jolt_serialize_u64(value) else {
            let _ = JoltProver.getLastError()
            throw JoltFFIError.unknownError
        }

        defer { jolt_serialized_free(handle) }

        var dataPtr: UnsafePointer<UInt8>?
        var dataLen: UInt = 0

        guard jolt_serialized_get_data(handle, &dataPtr, &dataLen) == 0,
              let ptr = dataPtr else {
            throw JoltFFIError.unknownError
        }

        return Data(bytes: ptr, count: Int(dataLen))
    }

    /// Serialize an array of UInt64 values using postcard
    static func serialize(_ values: [UInt64]) throws -> Data {
        guard let handle = values.withUnsafeBufferPointer({ bufferPtr in
            jolt_serialize_u64_array(bufferPtr.baseAddress, UInt(bufferPtr.count))
        }) else {
            let _ = JoltProver.getLastError()
            throw JoltFFIError.unknownError
        }

        defer { jolt_serialized_free(handle) }

        var dataPtr: UnsafePointer<UInt8>?
        var dataLen: UInt = 0

        guard jolt_serialized_get_data(handle, &dataPtr, &dataLen) == 0,
              let ptr = dataPtr else {
            throw JoltFFIError.unknownError
        }

        return Data(bytes: ptr, count: Int(dataLen))
    }

    /// Serialize a single UInt32 value using postcard
    static func serialize(_ value: UInt32) throws -> Data {
        guard let handle = jolt_serialize_u32(value) else {
            let _ = JoltProver.getLastError()
            throw JoltFFIError.unknownError
        }

        defer { jolt_serialized_free(handle) }

        var dataPtr: UnsafePointer<UInt8>?
        var dataLen: UInt = 0

        guard jolt_serialized_get_data(handle, &dataPtr, &dataLen) == 0,
              let ptr = dataPtr else {
            throw JoltFFIError.unknownError
        }

        return Data(bytes: ptr, count: Int(dataLen))
    }

    /// Serialize a String using postcard
    static func serialize(_ string: String) throws -> Data {
        guard let utf8Data = string.data(using: .utf8) else {
            throw JoltFFIError.unknownError
        }

        let handle = utf8Data.withUnsafeBytes { bufferPtr -> OpaquePointer? in
            guard let baseAddress = bufferPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return nil
            }
            return jolt_serialize_string(baseAddress, UInt(bufferPtr.count))
        }

        guard let handle = handle else {
            let _ = JoltProver.getLastError()
            throw JoltFFIError.unknownError
        }

        defer { jolt_serialized_free(handle) }

        var dataPtr: UnsafePointer<UInt8>?
        var dataLen: UInt = 0

        guard jolt_serialized_get_data(handle, &dataPtr, &dataLen) == 0,
              let ptr = dataPtr else {
            throw JoltFFIError.unknownError
        }

        return Data(bytes: ptr, count: Int(dataLen))
    }

    /// Concatenate multiple serialized values (for multi-parameter functions)
    static func concatenate(_ parts: Data...) -> Data {
        var result = Data()
        for part in parts {
            result.append(part)
        }
        return result
    }
}

// MARK: - Convenience Extensions

extension JoltProver {
    /// Generate a prover from ELF file with postcard-serialized inputs
    /// - Parameters:
    ///   - elfPath: Path to the ELF file
    ///   - u64Input: Single u64 input to serialize with postcard
    func generateProver(elfPath: String, u64Input: UInt64) throws {
        let serializedInput = try PostcardSerializer.serialize(u64Input)
        try generateProver(elfPath: elfPath, inputs: serializedInput)
    }

    /// Generate a prover from ELF file with postcard-serialized array inputs
    /// - Parameters:
    ///   - elfPath: Path to the ELF file
    ///   - u64ArrayInput: Array of u64 inputs to serialize with postcard
    func generateProver(elfPath: String, u64ArrayInput: [UInt64]) throws {
        let serializedInput = try PostcardSerializer.serialize(u64ArrayInput)
        try generateProver(elfPath: elfPath, inputs: serializedInput)
    }

    /// Generate a prover from ELF file with postcard-serialized string input
    /// - Parameters:
    ///   - elfPath: Path to the ELF file
    ///   - stringInput: String input to serialize with postcard
    func generateProver(elfPath: String, stringInput: String) throws {
        let serializedInput = try PostcardSerializer.serialize(stringInput)
        try generateProver(elfPath: elfPath, inputs: serializedInput)
    }
}
