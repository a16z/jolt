import Foundation
import SwiftUI

@MainActor
class JoltProverViewModel: ObservableObject {
    @Published var preprocessingFileName: String?
    @Published var elfFileName: String?
    @Published var showPreprocessingPicker = false
    @Published var showElfPicker = false
    @Published var statusMessage: String?
    @Published var hasError = false
    @Published var isProcessing = false

    private var preprocessingPath: String?
    private var elfPath: String?

    var canGenerateProof: Bool {
        preprocessingPath != nil && elfPath != nil && !isProcessing
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

    func generateProof() {
        guard let preprocessingPath = preprocessingPath,
              let elfPath = elfPath else {
            statusMessage = "Please select both preprocessing and ELF files"
            hasError = true
            return
        }

        isProcessing = true
        hasError = false
        statusMessage = "Initializing prover..."

        Task {
            do {
                // Create output path
                let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let proofPath = documentsPath.appendingPathComponent("proof.bin").path

                // Initialize prover
                statusMessage = "Loading preprocessing..."
                let prover = try JoltProver(preprocessingPath: preprocessingPath)

                // Generate prover from ELF
                statusMessage = "Generating prover from ELF..."
                try prover.generateProver(elfPath: elfPath)

                // Generate proof
                statusMessage = "Generating proof (this may take a while)..."
                try prover.prove(outputPath: proofPath)

                // Success
                statusMessage = "Proof generated successfully!\nSaved to: \(proofPath)"
                hasError = false
            } catch let error as JoltFFIError {
                statusMessage = error.localizedDescription
                hasError = true
            } catch {
                statusMessage = "Unexpected error: \(error.localizedDescription)"
                hasError = true
            }

            isProcessing = false
        }
    }
}
