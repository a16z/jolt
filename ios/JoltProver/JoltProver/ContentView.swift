import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = JoltProverViewModel()

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
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
                .padding(.bottom, 20)

                // Scrollable content area
                ScrollView {
                    VStack(spacing: 20) {
                        // Demo Proof Button
                        Button(action: {
                            viewModel.generateDemoProof()
                        }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(0.8)
                                } else {
                                    Image(systemName: "sparkles")
                                }

                                Text("Generate Fibonacci Demo Proof")
                                    .fontWeight(.semibold)
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.8)
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                            .background(viewModel.canGenerateDemoProof ? Color.green : Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        }
                        .disabled(!viewModel.canGenerateDemoProof)
                        .padding(.horizontal)
                        
                        // Divider
                        HStack(spacing: 12) {
                            Rectangle()
                                .fill(Color.secondary.opacity(0.3))
                                .frame(height: 1)
                            
                            Text("OR")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Rectangle()
                                .fill(Color.secondary.opacity(0.3))
                                .frame(height: 1)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)

                        // File Selection Section
                        VStack(spacing: 16) {
                            Text("Use custom files")
                                .font(.headline)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            
                            // Preprocessing File Picker
                            FilePickerButton(
                                title: "Select Preprocessing File",
                                selectedFileName: viewModel.preprocessingFileName,
                                action: {
                                    print("DEBUG: Setting activeFilePicker to .preprocessing")
                                    viewModel.activeFilePicker = .preprocessing
                                }
                            )

                            // ELF File Picker
                            FilePickerButton(
                                title: "Select ELF File",
                                selectedFileName: viewModel.elfFileName,
                                action: {
                                    print("DEBUG: Setting activeFilePicker to .elf")
                                    viewModel.activeFilePicker = .elf
                                }
                            )
                        }
                        .padding(.horizontal)

                        // Generate Custom Proof Button
                        Button(action: {
                            viewModel.generateCustomProof()
                        }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(0.8)
                                } else {
                                    Image(systemName: "lock.shield.fill")
                                }

                                Text("Generate Custom Proof")
                                    .fontWeight(.semibold)
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.8)
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                            .background(viewModel.canGenerateProof ? Color.blue : Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        }
                        .disabled(!viewModel.canGenerateProof)
                        .padding(.horizontal)

                        // Status Message
                        if let statusMessage = viewModel.statusMessage {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: viewModel.hasError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                        .foregroundColor(viewModel.hasError ? .red : .green)

                                    Text(viewModel.hasError ? "Error" : "Success")
                                        .fontWeight(.semibold)
                                }

                                Text(statusMessage)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                // Share button - only show on success when proof file exists
                                if !viewModel.hasError, let _ = viewModel.proofFileURL {
                                    Button(action: {
                                        viewModel.showShareSheet = true
                                    }) {
                                        HStack {
                                            Image(systemName: "square.and.arrow.up")
                                            Text("Share Proof File")
                                                .fontWeight(.medium)
                                        }
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 10)
                                        .background(Color.blue)
                                        .foregroundColor(.white)
                                        .cornerRadius(8)
                                    }
                                    .padding(.top, 4)
                                }
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(viewModel.hasError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                            )
                            .padding(.horizontal)
                        }
                        
                        // Spacer to push footer down
                        Spacer(minLength: 40)
                    }
                    .padding(.vertical)
                }

                // Info Footer - fixed at bottom
                VStack(spacing: 4) {
                    Text("Proof files are saved to app Documents")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("Output: proof.bin")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .fontWeight(.medium)
                }
                .padding(.vertical, 16)
                .background(Color(UIColor.systemBackground))
            }
            .navigationBarTitleDisplayMode(.inline)
        }
        .fileImporter(
            isPresented: Binding(
                get: {
                    let isPresented = viewModel.activeFilePicker != nil
                    if isPresented {
                        // Capture the type before it gets cleared
                        viewModel.lastActiveFilePicker = viewModel.activeFilePicker
                    }
                    print("DEBUG: fileImporter isPresented = \(isPresented), activeFilePicker = \(String(describing: viewModel.activeFilePicker))")
                    return isPresented
                },
                set: { newValue in
                    print("DEBUG: fileImporter set to \(newValue)")
                    if !newValue {
                        viewModel.activeFilePicker = nil
                        // Don't clear lastActiveFilePicker here - we need it in the completion
                    }
                }
            ),
            allowedContentTypes: (viewModel.activeFilePicker ?? viewModel.lastActiveFilePicker) == .preprocessing ?
                [UTType(filenameExtension: "bin")!] :
                [UTType(filenameExtension: "elf")!],
            allowsMultipleSelection: false
        ) { result in
            // Use the captured value instead of the current one
            let pickerType = viewModel.lastActiveFilePicker
            print("DEBUG: fileImporter completion called with lastActiveFilePicker = \(String(describing: pickerType))")
            
            switch pickerType {
            case .preprocessing:
                print("DEBUG: Handling preprocessing file")
                viewModel.handlePreprocessingFileSelection(result: result)
            case .elf:
                print("DEBUG: Handling ELF file")
                viewModel.handleElfFileSelection(result: result)
            case .none:
                print("DEBUG: lastActiveFilePicker is none")
                break
            }
            
            // Clear the cached value after handling
            viewModel.lastActiveFilePicker = nil
        }.sheet(isPresented: $viewModel.showShareSheet) {
            if let proofFileURL = viewModel.proofFileURL {
                ShareSheet(items: [proofFileURL])
            }
        }
    }
}

struct FilePickerButton: View {
    let title: String
    let selectedFileName: String?
    let action: () -> Void

    var body: some View {
        Button(action: {
            print("DEBUG: Button tapped for \(title)")
            action()
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

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: items,
            applicationActivities: nil
        )
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    ContentView()
}
