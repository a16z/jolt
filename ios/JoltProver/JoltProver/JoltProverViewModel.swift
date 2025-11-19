import Foundation
import SwiftUI
import OSLog

struct ProofStats {
    let inputDescription: String
    let cycles: String?
    let virtualCycles: String?
    let provingTime: String?
    let throughput: String?
}

enum ProofInput: Hashable {
    case u64(UInt64)
    case string(String)
    case u64Array([UInt64])
    case data(Data)

    func formattedDescription() -> String {
        switch self {
        case .u64(let value):
            return "Input: \(value)"
        case .string(let value):
            return "Input: \"\(value)\""
        case .u64Array(let value):
            return "Input: [\(value.map { String($0) }.joined(separator: ", "))]"
        case .data(let value):
            return "Input: \(value.count) bytes"
        }
    }
}

struct DemoProof: Identifiable, Hashable {
    let id: String
    let name: String
    let preprocessingFile: String
    let elfFile: String
    let input: ProofInput

    var preprocessingPath: String? {
        Bundle.main.path(forResource: preprocessingFile, ofType: "bin")
    }

    var elfPath: String? {
        Bundle.main.path(forResource: elfFile, ofType: "elf")
    }

    var isAvailable: Bool {
        preprocessingPath != nil && elfPath != nil
    }
}

@MainActor
class JoltProverViewModel: ObservableObject {
    @Published var preprocessingFileName: String?
    @Published var elfFileName: String?
    @Published var activeFilePicker: FilePickerType?
    @Published var statusMessage: String?
    @Published var hasError = false
    @Published var isProcessing = false
    @Published var proofFileURL: URL?
    @Published var showShareSheet = false
    @Published var selectedDemo: DemoProof
    @Published var proofStats: ProofStats?
    @Published var isRunningDemo: Bool = false

    private var preprocessingPath: String?
    private var elfPath: String?

    var lastActiveFilePicker: FilePickerType?

    enum FilePickerType {
        case preprocessing
        case elf
    }

    // Available demo proofs
    let availableDemos = [
        DemoProof(
            id: "fibonacci",
            name: "Fibonacci",
            preprocessingFile: "fibonacci-preprocessing",
            elfFile: "fibonacci",
            input: .u64(80000)
        ),
        DemoProof(
            id: "sha3-chain",
            name: "SHA-3-chain",
            preprocessingFile: "sha3-chain-preprocessing",
            elfFile: "sha3-chain",
            input: .data(try! PostcardSerializer.concatenate(
                Data(repeating: 0x42, count: 32), // input
                PostcardSerializer.serialize(30 as UInt32) // iterations
            ))
        ),
        DemoProof(
            id: "ecdsa-sign",
            name: "ECDSA Sign",
            preprocessingFile: "ecdsa-sign-preprocessing",
            elfFile: "ecdsa-sign",
            input: .data(PostcardSerializer.concatenate(
                Data(repeating: 0x42, count: 32), // privateKey
                Data(repeating: 0x10, count: 32) // MessageHash
            ))
        )
    ]

    init() {
        self.selectedDemo = availableDemos[0]
    }

    var canGenerateProof: Bool {
        !isProcessing && preprocessingPath != nil && elfPath != nil
    }

    var canGenerateDemoProof: Bool {
        !isProcessing && selectedDemo.isAvailable
    }

    var hasCustomFiles: Bool {
        preprocessingPath != nil && elfPath != nil
    }

    var shouldShowDemoSpinner: Bool {
        isProcessing && isRunningDemo
    }

    var shouldShowCustomSpinner: Bool {
        isProcessing && !isRunningDemo
    }

    func handlePreprocessingFileSelection(result: Result<[URL], Error>) {
        do {
            let urls = try result.get()
            guard let url = urls.first else { return }

            // Start accessing a security-scoped resource
            guard url.startAccessingSecurityScopedResource() else {
                statusMessage = "Cannot access file"
                hasError = true
                return
            }
            defer { url.stopAccessingSecurityScopedResource() }

            // Copy file to app's Documents directory
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let destURL = documentsPath.appendingPathComponent("preprocessing.bin")

            // Remove existing file if present
            try? FileManager.default.removeItem(at: destURL)

            // Copy file
            try FileManager.default.copyItem(at: url, to: destURL)

            preprocessingPath = destURL.path
            preprocessingFileName = url.lastPathComponent
            statusMessage = nil
            hasError = false
        } catch {
            statusMessage = "Failed to load preprocessing file: \(error.localizedDescription)"
            hasError = true
        }
    }

    func handleElfFileSelection(result: Result<[URL], Error>) {
        do {
            let urls = try result.get()
            guard let url = urls.first else { return }

            // Start accessing a security-scoped resource
            guard url.startAccessingSecurityScopedResource() else {
                statusMessage = "Cannot access file"
                hasError = true
                return
            }
            defer { url.stopAccessingSecurityScopedResource() }

            // Copy file to app's Documents directory
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let destURL = documentsPath.appendingPathComponent("guest.elf")

            // Remove existing file if present
            try? FileManager.default.removeItem(at: destURL)

            // Copy file
            try FileManager.default.copyItem(at: url, to: destURL)

            elfPath = destURL.path
            elfFileName = url.lastPathComponent
            statusMessage = nil
            hasError = false
        } catch {
            statusMessage = "Failed to load ELF file: \(error.localizedDescription)"
            hasError = true
        }
    }

    func generateDemoProof() {
        guard let preprocessingPath = selectedDemo.preprocessingPath,
              let elfPath = selectedDemo.elfPath else {
            statusMessage = "Demo files not found for \(selectedDemo.name)"
            hasError = true
            return
        }

        isRunningDemo = true
        isProcessing = true
        generateProofInternal(
            preprocessingPath: preprocessingPath,
            elfPath: elfPath,
            input: selectedDemo.input,
            demoName: selectedDemo.name
        )
    }

    func generateCustomProof() {
        guard let preprocessingPath = preprocessingPath,
              let elfPath = elfPath else {
            statusMessage = "Please select both preprocessing and ELF files"
            hasError = true
            return
        }

        isRunningDemo = false
        isProcessing = true
        generateProofInternal(
            preprocessingPath: preprocessingPath,
            elfPath: elfPath,
            input: .u64(80000), // Default input for custom proofs
            demoName: nil
        )
    }

    private func generateProofInternal(preprocessingPath: String, elfPath: String, input: ProofInput, demoName: String?) {
        // isProcessing is already set by the caller
        hasError = false
        proofStats = nil

        let inputDesc = input.formattedDescription()
        statusMessage = demoName != nil ? "Initializing \(demoName!) demo prover...\n\(inputDesc)" : "Initializing prover...\n\(inputDesc)"

        Task {
            do {
                // Reset Dory global state to prevent dimension mismatch when switching between demos
                reset_dory()

                // Create output path
                let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let proofURL = documentsPath.appendingPathComponent("proof.bin")

                // Initialize prover
                statusMessage = "Loading preprocessing...\n\(inputDesc)"
                let prover = try JoltProver(preprocessingPath: preprocessingPath)

                // Generate prover from ELF with appropriate input type
                statusMessage = "Generating prover from ELF...\n\(inputDesc)"
                switch input {
                case .u64(let value):
                    try prover.generateProver(elfPath: elfPath, u64Input: value)
                case .string(let value):
                    try prover.generateProver(elfPath: elfPath, stringInput: value)
                case .u64Array(let value):
                    try prover.generateProver(elfPath: elfPath, u64ArrayInput: value)
                case .data(let value):
                    try prover.generateProver(elfPath: elfPath, inputs: value)
                }

                // Generate proof
                statusMessage = "Generating proof (this may take a while)...\n\(inputDesc)"
                let startTime = Date()
                try prover.prove(outputPath: proofURL.path)
                let endTime = Date()
                let duration = endTime.timeIntervalSince(startTime)

                // Success - capture logs and create stats
                proofFileURL = proofURL
                let logs = captureLogs()
                proofStats = ProofStats(
                    inputDescription: inputDesc,
                    cycles: logs.cycles,
                    virtualCycles: logs.virtualCycles,
                    provingTime: String(format: "%.1fs", duration),
                    throughput: logs.throughput
                )
                statusMessage = formatSuccessMessage(stats: proofStats!)
                hasError = false
            } catch let error as JoltFFIError {
                statusMessage = error.localizedDescription
                hasError = true
                proofFileURL = nil
            } catch {
                statusMessage = "Unexpected error: \(error.localizedDescription)"
                hasError = true
                proofFileURL = nil
            }

            isProcessing = false
        }
    }

    // MARK: - Helper Functions

    /// Capture logs from Rust tracing output
    /// Parses OSLog for cycle counts and throughput statistics
    private func captureLogs() -> (cycles: String?, virtualCycles: String?, throughput: String?) {
        guard #available(iOS 15.0, *) else {
            return (nil, nil, nil)
        }

        do {
            let logStore = try OSLogStore(scope: .currentProcessIdentifier)
            let position = logStore.position(timeIntervalSinceEnd: -60) // Last 60 seconds

            var cycles: String?
            var virtualCycles: String?
            var throughput: String?

            let entries = try logStore.getEntries(at: position)

            for entry in entries {
                guard let logEntry = entry as? OSLogEntryLog else { continue }
                let message = logEntry.composedMessage

                // Parse: "880005 RV64IMAC cycles, 960007 virtual cycles"
                if message.contains("RV64IMAC cycles") && message.contains("virtual cycles") {
                    let components = message.components(separatedBy: ",")
                    if components.count >= 2 {
                        // Extract cycle count
                        if let cycleMatch = components[0].components(separatedBy: " ").first(where: { Int($0) != nil }) {
                            cycles = cycleMatch
                        }
                        // Extract virtual cycle count
                        if let virtualMatch = components[1].components(separatedBy: " ").first(where: { Int($0) != nil }) {
                            virtualCycles = virtualMatch
                        }
                    }
                }

                // Parse: "Proved in 49.3s (19.5 kHz / padded 21.3 kHz)"
                if message.contains("Proved in") && message.contains("kHz") {
                    // Extract the throughput part in parentheses
                    if let parenStart = message.firstIndex(of: "("),
                       let parenEnd = message.firstIndex(of: ")") {
                        let throughputStr = String(message[message.index(after: parenStart)..<parenEnd])
                        throughput = throughputStr
                    }
                }
            }

            return (cycles: cycles, virtualCycles: virtualCycles, throughput: throughput)
        } catch {
            print("Failed to capture logs: \(error)")
            return (nil, nil, nil)
        }
    }

    /// Format success message with proof statistics
    private func formatSuccessMessage(stats: ProofStats) -> String {
        var message = "Proof generated successfully!"

        if let cycles = stats.cycles, let virtualCycles = stats.virtualCycles {
            message += "\n\n\(cycles) RV64IMAC cycles\n\(virtualCycles) virtual cycles"
        }

        if let provingTime = stats.provingTime {
            message += "\n\nProving time: \(provingTime)"
        }

        if let throughput = stats.throughput {
            message += "\n\(throughput)"
        }

        return message
    }
}
