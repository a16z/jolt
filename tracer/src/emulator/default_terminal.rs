use super::terminal::Terminal;

/// Standard `Terminal`.
pub struct DefaultTerminal {
    input_data: Vec<u8>,
    output_data: Vec<u8>,
}

impl DefaultTerminal {
    pub fn new() -> Self {
        DefaultTerminal {
            input_data: vec![],
            output_data: vec![],
        }
    }
}

impl Terminal for DefaultTerminal {
    fn put_byte(&mut self, value: u8) {
        self.output_data.push(value);
    }

    fn get_input(&mut self) -> u8 {
        match !self.input_data.is_empty() {
            true => self.input_data.remove(0),
            false => 0,
        }
    }

    fn put_input(&mut self, value: u8) {
        self.input_data.push(value);
    }

    fn get_output(&mut self) -> u8 {
        match !self.output_data.is_empty() {
            true => self.output_data.remove(0),
            false => 0,
        }
    }
}
