import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = JoltProverViewModel()

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Text("Jolt zkVM Prover")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text("Generate zero-knowledge proofs on iOS")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 20)

                Spacer()

                // File Selection Section
                VStack(spacing: 16) {
                    // Preprocessing File Picker
                    FilePickerButton(
                        title: "Select Preprocessing File",
                        selectedFileName: viewModel.preprocessingFileName,
                        isPresented: $viewModel.showPreprocessingPicker
                    )

                    // ELF File Picker
                    FilePickerButton(
                        title: "Select ELF File",
                        selectedFileName: viewModel.elfFileName,
                        isPresented: $viewModel.showElfPicker
                    )
                }
                .padding(.horizontal)

                // Generate Proof Button
                Button(action: {
                    viewModel.generateProof()
                }) {
                    HStack {
                        if viewModel.isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "lock.shield.fill")
                        }

                        Text(viewModel.isProcessing ? "Generating Proof..." : "Generate Proof")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.canGenerateProof ? Color.blue : Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(!viewModel.canGenerateProof || viewModel.isProcessing)
                .padding(.horizontal)

                // Status Message
                if let statusMessage = viewModel.statusMessage {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: viewModel.hasError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                .foregroundColor(viewModel.hasError ? .red : .green)

                            Text(viewModel.hasError ? "Error" : "Success")
                                .fontWeight(.semibold)
                        }

                        Text(statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(viewModel.hasError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    )
                    .padding(.horizontal)
                }

                Spacer()

                // Info Footer
                VStack(spacing: 4) {
                    Text("Proof files are saved to app Documents")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("Output: proof.bin")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .fontWeight(.medium)
                }
                .padding(.bottom, 20)
            }
            .navigationBarTitleDisplayMode(.inline)
        }
        .fileImporter(
            isPresented: $viewModel.showPreprocessingPicker,
            allowedContentTypes: [.data, .item],
            allowsMultipleSelection: false
        ) { result in
            viewModel.handlePreprocessingFileSelection(result: result)
        }
        .fileImporter(
            isPresented: $viewModel.showElfPicker,
            allowedContentTypes: [.data, .item],
            allowsMultipleSelection: false
        ) { result in
            viewModel.handleElfFileSelection(result: result)
        }
    }
}

struct FilePickerButton: View {
    let title: String
    let selectedFileName: String?
    @Binding var isPresented: Bool

    var body: some View {
        Button(action: {
            isPresented = true
        }) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .fontWeight(.medium)

                    if let fileName = selectedFileName {
                        Text(fileName)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("No file selected")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                Image(systemName: selectedFileName != nil ? "checkmark.circle.fill" : "folder")
                    .foregroundColor(selectedFileName != nil ? .green : .blue)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color(.systemGray6))
            )
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    ContentView()
}
