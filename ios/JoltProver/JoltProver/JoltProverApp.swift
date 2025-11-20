import SwiftUI
import os.log

@main
struct JoltProverApp: App {
    init() {
        // Copy bundled Dory URS files to cache directory on first launch
        setupDoryCache()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }

    /// Copy bundled Dory URS files to the cache directory
    /// This avoids generating them on first proof, which can be slow
    private func setupDoryCache() {
        guard let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            print("Failed to get cache directory")
            return
        }

        let doryDir = cacheDir.appendingPathComponent("dory")

        // Create dory directory if it doesn't exist
        try? FileManager.default.createDirectory(at: doryDir, withIntermediateDirectories: true)

        let ursFiles = [
            "dory_32.urs"
        ]

        for filename in ursFiles {
            let destURL = doryDir.appendingPathComponent(filename)

            // Skip if file already exists
            guard !FileManager.default.fileExists(atPath: destURL.path) else {
                continue
            }

            // Copy from bundle
            guard let bundledURL = Bundle.main.url(forResource: filename.replacingOccurrences(of: ".urs", with: ""), withExtension: "urs") else {
                print("Bundled URS file not found: \(filename)")
                continue
            }

            do {
                try FileManager.default.copyItem(at: bundledURL, to: destURL)
                print("Copied \(filename) to cache")
            } catch {
                print("Failed to copy \(filename): \(error)")
            }
        }
    }
}
