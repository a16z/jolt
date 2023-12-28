use std::fs::File;
use std::io::{Read, Write};
use std::io::BufReader;
use std::io::BufWriter;
use std::path::PathBuf;

pub fn execute_template(circuit_template_location: &PathBuf, circuit_destination: &PathBuf, num_steps: usize, prog_start: usize) {
    // Load the template
    let mut file = BufReader::new(File::open(circuit_template_location).expect("Could not open circuit template"));
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Could not read circuit template");

    // Check and replace placeholders
    if contents.contains("<NUM_STEPS>") && contents.contains("<PROG_START_ADDR>") {
        contents = contents.replace("<NUM_STEPS>", &num_steps.to_string());
        contents = contents.replace("<PROG_START_ADDR>", &prog_start.to_string());
    } else {
        panic!("Could not find placeholders in the template");
    }

    // Write to destination
    let path = circuit_destination;
    std::fs::create_dir_all(path.parent().unwrap()).expect("Failed to create directories");
    let mut file = BufWriter::new(File::create(&path).expect("Could not create circuit destination file"));
    file.write_all(contents.as_bytes()).expect("Could not write to circuit destination file");
}
