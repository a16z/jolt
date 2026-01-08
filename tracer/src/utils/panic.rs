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
    let panic_info = if !call_stack.is_empty() && emulator_state.elf_path.is_some() {
        if let Ok(loader) = addr2line::Loader::new(emulator_state.elf_path.as_ref().unwrap()) {
            let last_frame = call_stack.back().unwrap();
            resolve_frame(&loader, last_frame)
                .map(|resolved| (resolved.function_name, resolved.location))
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
