use crate::emulator::cpu::get_register_name;
use crate::emulator::Emulator;
use crate::instruction::format::NormalizedOperands;
use common::constants::{REGISTER_COUNT, RISCV_REGISTER_COUNT};

/// Represents a single function call for stack trace
#[derive(Clone, Copy, Debug)]
pub struct CallFrame {
    pub call_site: u64,
    /// register snapshot
    pub x: [i64; REGISTER_COUNT as usize],
    pub operands: NormalizedOperands,
    /// cycle count at the time of call
    pub cycle_count: usize,
}

#[derive(Debug)]
pub struct ResolvedFrame {
    pub function_name: String,
    pub location: Option<SourceLocation>,
    pub inlined_frames: Vec<(String, Option<SourceLocation>)>,
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

impl SourceLocation {
    pub fn fmt_short(&self) -> String {
        match (self.line, self.column) {
            (Some(line), Some(col)) => format!("{}:{}:{}", self.file, line, col),
            (Some(line), None) => format!("{}:{}", self.file, line),
            _ => self.file.clone(),
        }
    }
}

pub fn resolve_frame(loader: &addr2line::Loader, frame: &CallFrame) -> Option<ResolvedFrame> {
    let addr = frame.call_site;

    if let Ok(mut frames) = loader.find_frames(addr) {
        let mut primary_frame = None;
        let mut inlined = vec![];

        while let Ok(Some(frame)) = frames.next() {
            let name = frame
                .function
                .and_then(|f| f.demangle().ok().map(|s| s.to_string()))
                .unwrap_or_else(|| "<unknown>".to_string());

            let location = frame.location.map(|loc| SourceLocation {
                file: shorten_path(loc.file.unwrap_or("??")),
                line: loc.line,
                column: loc.column,
            });

            if primary_frame.is_none() {
                primary_frame = Some((name, location));
            } else {
                inlined.push((name, location));
            }
        }

        if let Some((name, location)) = primary_frame {
            return Some(ResolvedFrame {
                function_name: name,
                location,
                inlined_frames: inlined,
            });
        }
    }

    // Fallback to just symbol name
    if let Some(sym_name) = loader.find_symbol(addr) {
        let demangled = addr2line::demangle_auto(std::borrow::Cow::Borrowed(sym_name), None);
        return Some(ResolvedFrame {
            function_name: demangled.to_string(),
            location: None,
            inlined_frames: vec![],
        });
    }

    None
}

fn shorten_path(path: &str) -> String {
    const MAX_LENGTH: usize = 50; // Adjust as needed

    if path.len() <= MAX_LENGTH {
        return path.to_string();
    }

    let components: Vec<&str> = path.split('/').collect();

    let filename = components.last().unwrap_or(&"");

    let mut result_components = vec![*filename];
    let mut total_len = filename.len();

    for component in components.iter().rev().skip(1) {
        let new_len = total_len + component.len() + 1; // +1 for '/'
        if new_len > MAX_LENGTH - 4 {
            // -4 for ".../""
            break;
        }
        result_components.insert(0, *component);
        total_len = new_len;
    }

    // Add ellipsis if we truncated
    if result_components.len() < components.len() {
        format!(".../{}", result_components.join("/"))
    } else {
        result_components.join("/")
    }
}

pub fn display_panic_backtrace(emulator_state: &Emulator) {
    let cpu = emulator_state.get_cpu();
    let call_stack = cpu.get_call_stack();

    if call_stack.is_empty() || emulator_state.elf_path.is_none() {
        println!("  <no backtrace available>");
        println!(
            "note: run `trace_and_analyze` with `JOLT_BACKTRACE=1` environment variable to enable backtraces"
        );
        return;
    }

    let loader = match emulator_state
        .elf_path
        .as_ref()
        .and_then(|path| addr2line::Loader::new(path).ok())
    {
        Some(loader) => loader,
        None => {
            println!("  <failed to load symbols>");
            return;
        }
    };

    // Get panic location from most recent frame
    let panic_info = if !call_stack.is_empty() {
        if let Some(elf_path) = emulator_state.elf_path.as_ref() {
            if let Ok(loader) = addr2line::Loader::new(elf_path) {
                let last_frame = call_stack.back().unwrap();
                resolve_frame(&loader, last_frame)
                    .map(|resolved| (resolved.function_name, resolved.location))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    if let Some((func, Some(loc))) = &panic_info {
        println!(
            "Guest Program panicked in \"{}\" at \"{}\"",
            func,
            loc.fmt_short()
        );
    }

    println!("stack backtrace:");

    let full_backtrace = std::env::var("JOLT_BACKTRACE")
        .map(|v| v.eq_ignore_ascii_case("full"))
        .unwrap_or(false);

    for (frame_num, frame) in call_stack.iter().rev().enumerate() {
        print!("{:4}: {:#08x} - ", frame_num, frame.call_site);

        if let Some(resolved) = resolve_frame(&loader, frame) {
            print_resolved_frame(&resolved);
        } else {
            println!("<unknown>");
        }

        if full_backtrace {
            print_extended_frame_info(frame);
        }
    }

    if panic_info.is_none() {
        println!(
            "note: run with `JOLT_BACKTRACE=1` environment variable to symbolize the backtrace"
        );
    } else if !full_backtrace {
        println!("note: run with `JOLT_BACKTRACE=full` environment variable to display extended emulator state info");
    }
}

fn print_resolved_frame(resolved: &ResolvedFrame) {
    print!("{}", resolved.function_name);
    if let Some(loc) = &resolved.location {
        println!();
        println!("                               at {}", loc.fmt_short());
    } else {
        println!();
    }

    for (name, location) in &resolved.inlined_frames {
        println!("                   {name}");
        if let Some(loc) = location {
            println!("                               at {}", loc.fmt_short());
        }
    }
}

fn print_extended_frame_info(frame: &CallFrame) {
    let mut regs = vec![];
    for i in 0..RISCV_REGISTER_COUNT {
        let i = i as usize;
        if frame.x[i] != 0 || i == 2 {
            // Always show sp
            regs.push(format!("{}={:#x}", get_register_name(i), frame.x[i]));
        }
    }
    if !regs.is_empty() {
        println!("                   registers: {}", regs.join(", "));
    }
    println!("                   cycle: {}", frame.cycle_count);
    println!();
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only assertions"
)]
mod tests {
    use super::*;
    use crate::emulator::default_terminal::DefaultTerminal;
    use crate::emulator::elf_analyzer::test_elf::{build_elf64, TestSymbol};

    #[test]
    fn fmt_short_includes_only_the_known_position_parts() {
        let with_both = SourceLocation {
            file: "src/lib.rs".to_string(),
            line: Some(10),
            column: Some(4),
        };
        assert_eq!(with_both.fmt_short(), "src/lib.rs:10:4");

        let line_only = SourceLocation {
            file: "src/lib.rs".to_string(),
            line: Some(10),
            column: None,
        };
        assert_eq!(line_only.fmt_short(), "src/lib.rs:10");

        let file_only = SourceLocation {
            file: "src/lib.rs".to_string(),
            line: None,
            column: None,
        };
        assert_eq!(file_only.fmt_short(), "src/lib.rs");
    }

    #[test]
    fn shorten_path_keeps_trailing_components_within_the_budget() {
        // Short paths pass through untouched
        assert_eq!(shorten_path("src/lib.rs"), "src/lib.rs");

        // 57 chars: the head component no longer fits the 46-char budget
        let long = "aaaaaaaaaa/bbbbbbbbbb/cccccccccc/dddddddddd/eeeeeeeeee.rs";
        assert_eq!(
            shorten_path(long),
            ".../bbbbbbbbbb/cccccccccc/dddddddddd/eeeeeeeeee.rs"
        );
    }

    fn write_test_elf() -> std::path::PathBuf {
        let elf = build_elf64(
            &[0x0010_0093, 0x0000_006f],
            &[TestSymbol {
                name: "_start",
                value: 0x8000_0000,
                info: 0x12, // GLOBAL | FUNC
                size: 8,
            }],
        );
        let path =
            std::env::temp_dir().join(format!("jolt-panic-test-{}-guest.elf", std::process::id()));
        std::fs::write(&path, elf).unwrap();
        path
    }

    #[test]
    fn resolve_frame_falls_back_to_the_symbol_table_without_debug_info() {
        let path = write_test_elf();
        let loader = addr2line::Loader::new(&path).expect("loader parses the ELF");
        let frame = CallFrame {
            call_site: 0x8000_0004, // inside _start (size 8)
            x: [0; REGISTER_COUNT as usize],
            operands: NormalizedOperands::default(),
            cycle_count: 3,
        };
        let resolved = resolve_frame(&loader, &frame).expect("symbol found");
        assert_eq!(resolved.function_name, "_start");
        assert!(resolved.location.is_none(), "fixture has no debug info");
        assert!(resolved.inlined_frames.is_empty());

        // An address outside every symbol resolves to nothing
        let stray = CallFrame {
            call_site: 0x9000_0000,
            ..frame
        };
        assert!(resolve_frame(&loader, &stray).is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn display_panic_backtrace_handles_all_loader_outcomes() {
        use crate::emulator::Emulator;

        // 1. No call stack and no ELF path: prints the "<no backtrace>" note.
        let emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        display_panic_backtrace(&emulator);

        // 2. An ELF path that is not a valid ELF: loader fails gracefully.
        let garbage = std::env::temp_dir().join(format!(
            "jolt-panic-test-{}-garbage.elf",
            std::process::id()
        ));
        std::fs::write(&garbage, b"not an elf").unwrap();
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.set_elf_path(&garbage);
        emulator
            .get_mut_cpu()
            .track_call(0x8000_0004, NormalizedOperands::default());
        display_panic_backtrace(&emulator);
        std::fs::remove_file(&garbage).ok();

        // 3. A valid ELF with a symbolizable frame walks the full print path.
        let path = write_test_elf();
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.set_elf_path(&path);
        emulator
            .get_mut_cpu()
            .track_call(0x8000_0004, NormalizedOperands::default());
        emulator
            .get_mut_cpu()
            .track_call(0x9000_0000, NormalizedOperands::default());
        display_panic_backtrace(&emulator);
        std::fs::remove_file(&path).ok();
    }
}
