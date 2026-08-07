use super::terminal::Terminal;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Standard `Terminal`.
#[derive(Default)]
pub struct DefaultTerminal {
    input_data: Vec<u8>,
    output_data: Vec<u8>,
}

impl Terminal for DefaultTerminal {
    fn put_byte(&mut self, value: u8) {
        self.output_data.push(value);
    }

    fn get_output(&mut self) -> u8 {
        match !self.output_data.is_empty() {
            true => self.output_data.remove(0),
            false => 0,
        }
    }

    fn put_input(&mut self, value: u8) {
        self.input_data.push(value);
    }

    fn get_input(&mut self) -> u8 {
        match !self.input_data.is_empty() {
            true => self.input_data.remove(0),
            false => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_are_independent_fifos_that_drain_to_zero() {
        let mut terminal = DefaultTerminal::default();
        terminal.put_byte(b'a');
        terminal.put_byte(b'b');
        terminal.put_input(b'x');

        assert_eq!(terminal.get_output(), b'a');
        assert_eq!(terminal.get_output(), b'b');
        assert_eq!(terminal.get_output(), 0, "drained buffer yields 0");
        assert_eq!(terminal.get_input(), b'x');
        assert_eq!(terminal.get_input(), 0);
    }
}
