import Foundation
import SwiftUI

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

    private var preprocessingPath: String?
    private var elfPath: String?
    
    // Use a regular property, not @Published
    var lastActiveFilePicker: FilePickerType?
    
    enum FilePickerType {
        case preprocessing
        case elf
    }
    
    // Built-in asset paths
    private let builtInPreprocessingPath: String? = {
        let path = Bundle.main.path(forResource: "fibonacci-guest-preprocessing", ofType: "bin")
        print("DEBUG: Built-in preprocessing path: \(path ?? "nil")")
        return path
    }()
    
    private let builtInElfPath: String? = {
        let path = Bundle.main.path(forResource: "fibonacci-guest", ofType: "elf")
        print("DEBUG: Built-in ELF path: \(path ?? "nil")")
        return path
    }()

    var canGenerateProof: Bool {
        !isProcessing && preprocessingPath != nil && elfPath != nil
    }
    
    var canGenerateDemoProof: Bool {
        !isProcessing && builtInPreprocessingPath != nil && builtInElfPath != nil
    }
    
    var hasCustomFiles: Bool {
        preprocessingPath != nil && elfPath != nil
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
        guard let preprocessingPath = builtInPreprocessingPath,
              let elfPath = builtInElfPath else {
            statusMessage = "Built-in demo files not found"
            hasError = true
            return
        }
        
        generateProofInternal(preprocessingPath: preprocessingPath, elfPath: elfPath, isDemo: true)
    }

    func generateCustomProof() {
        guard let preprocessingPath = preprocessingPath,
              let elfPath = elfPath else {
            statusMessage = "Please select both preprocessing and ELF files"
            hasError = true
            return
        }
        
        generateProofInternal(preprocessingPath: preprocessingPath, elfPath: elfPath, isDemo: false)
    }
    
    private func generateProofInternal(preprocessingPath: String, elfPath: String, isDemo: Bool) {
            isProcessing = true
            hasError = false
            statusMessage = isDemo ? "Initializing Fibonacci demo prover..." : "Initializing prover..."

        Task {
            do {
                // Create output path
                let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let proofURL = documentsPath.appendingPathComponent("proof.bin")

                // Initialize prover
                statusMessage = "Loading preprocessing..."
                let prover = try JoltProver(preprocessingPath: preprocessingPath)

                // Generate prover from ELF
                statusMessage = "Generating prover from ELF..."
                try prover.generateProver(elfPath: elfPath)

                // Generate proof
                statusMessage = "Generating proof (this may take a while)..."
                try prover.prove(outputPath: proofURL.path)

                // Success
                proofFileURL = proofURL
                statusMessage = "Proof generated successfully!\nSaved to: \(proofURL.path)"
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
}
