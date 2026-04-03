/// Thin indentation-aware string builder for MSL codegen.
///
/// Tracks indent depth and provides `line()` for static text and `linef()` for
/// formatted text, both auto-indented. Use the `msl!` macro as shorthand for
/// `linef`.
pub(crate) struct Msl {
    buf: String,
    depth: u8,
}

impl Msl {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: String::with_capacity(capacity),
            depth: 0,
        }
    }

    /// Create a writer pre-indented to `depth` levels.
    /// Useful for body snippets injected into an outer kernel at a known depth.
    pub fn new_at(capacity: usize, depth: u8) -> Self {
        Self {
            buf: String::with_capacity(capacity),
            depth,
        }
    }

    /// Indented line from a plain `&str`. No format escaping needed —
    /// braces are literal, so `w.line("if (x) {")` just works.
    pub fn line(&mut self, text: &str) {
        self.write_indent();
        self.buf.push_str(text);
        self.buf.push('\n');
    }

    /// Indented line with `format_args!` interpolation. Called via `msl!` macro.
    pub fn linef(&mut self, args: std::fmt::Arguments<'_>) {
        self.write_indent();
        std::fmt::Write::write_fmt(&mut self.buf, args).unwrap();
        self.buf.push('\n');
    }

    /// Raw append — no indent, no newline. For when you need manual control.
    pub fn raw(&mut self, text: &str) {
        self.buf.push_str(text);
    }

    /// Increase indent depth by one level (4 spaces).
    pub fn push(&mut self) {
        self.depth += 1;
    }

    /// Decrease indent depth by one level.
    pub fn pop(&mut self) {
        self.depth -= 1;
    }

    /// Emit a blank line (no indentation).
    pub fn blank(&mut self) {
        self.buf.push('\n');
    }

    /// Consume the writer and return the built string.
    pub fn into_string(self) -> String {
        self.buf
    }

    fn write_indent(&mut self) {
        for _ in 0..self.depth {
            self.buf.push_str("    ");
        }
    }
}

/// Shorthand for `$w.linef(format_args!(...))` — writes an indented formatted line.
macro_rules! msl {
    ($w:expr, $($arg:tt)*) => {
        $w.linef(format_args!($($arg)*))
    };
}
pub(crate) use msl;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_output() {
        let mut w = Msl::new(256);
        w.line("kernel void foo(");
        w.push();
        let k = 0;
        msl!(w, "device const Fr* input_{k} [[buffer({k})]],");
        w.pop();
        w.line(") {");
        w.push();
        w.line("uint n = params[0];");
        w.blank();
        w.line("if (n > 0) {");
        w.push();
        w.line("// body");
        w.pop();
        w.line("}");
        w.pop();
        w.line("}");

        let expected = "\
kernel void foo(
    device const Fr* input_0 [[buffer(0)]],
) {
    uint n = params[0];

    if (n > 0) {
        // body
    }
}
";
        assert_eq!(w.into_string(), expected);
    }
}
