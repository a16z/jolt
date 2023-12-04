use crate::emulator::terminal::Terminal;

const IER_RXINT_BIT: u8 = 0x1;
const IER_THREINT_BIT: u8 = 0x2;

const IIR_THR_EMPTY: u8 = 0x2;
const IIR_RD_AVAILABLE: u8 = 0x4;
const IIR_NO_INTERRUPT: u8 = 0x7;

const LSR_DATA_AVAILABLE: u8 = 0x1;
const LSR_THR_EMPTY: u8 = 0x20;

/// Emulates UART. Refer to the [specification](http://www.ti.com/lit/ug/sprugp1/sprugp1.pdf)
/// for the detail.
pub struct Uart {
    clock: u64,
    rbr: u8, // receiver buffer register
    thr: u8, // transmitter holding register
    ier: u8, // interrupt enable register
    iir: u8, // interrupt identification register
    lcr: u8, // line control register
    mcr: u8, // modem control register
    lsr: u8, // line status register
    scr: u8, // scratch,
    thre_ip: bool,
    interrupting: bool,
    terminal: Box<dyn Terminal>,
}

impl Uart {
    /// Creates a new `Uart`. Input/Output data is transferred via `Terminal`.
    pub fn new(terminal: Box<dyn Terminal>) -> Self {
        Uart {
            clock: 0,
            rbr: 0,
            thr: 0,
            ier: 0,
            iir: 0,
            lcr: 0,
            mcr: 0,
            lsr: LSR_THR_EMPTY,
            scr: 0,
            thre_ip: false,
            interrupting: false,
            terminal,
        }
    }

    /// Runs one cycle. `Uart` gets/puts input/output data via `Terminal`
    /// at certain timing.
    pub fn tick(&mut self) {
        self.clock = self.clock.wrapping_add(1);
        let mut rx_ip = false;

        // Reads input.
        // 0x38400 is just an arbitary number @TODO: Fix me
        if (self.clock % 0x38400) == 0 && self.rbr == 0 {
            let value = self.terminal.get_input();
            if value != 0 {
                self.rbr = value;
                self.lsr |= LSR_DATA_AVAILABLE;
                self.update_iir();
                if (self.ier & IER_RXINT_BIT) != 0 {
                    rx_ip = true;
                }
            }
        }

        // Writes output.
        // 0x10 is just an arbitary number @TODO: Fix me
        if (self.clock % 0x10) == 0 && self.thr != 0 {
            self.terminal.put_byte(self.thr);
            self.thr = 0;
            self.lsr |= LSR_THR_EMPTY;
            self.update_iir();
            if (self.ier & IER_THREINT_BIT) != 0 {
                self.thre_ip = true;
            }
        }

        if self.thre_ip || rx_ip {
            self.interrupting = true;
            self.thre_ip = false;
        } else {
            self.interrupting = false;
        }
    }

    /// Indicates whether an interrupt happens in the current cycle.
    /// Note: This behavior is for easily handling an interrupt as
    /// "Edge-triggered" from `Uart` module user.
    /// It doesn't seem to be mentioned in the UART specification
    /// whether interrupt should be "Edge-triggered" or "Level-triggered" but
    /// we implement it as "Edge-triggered" so far because it would support more
    /// drivers. I speculate some drivers assume "Edge-triggered" interrupt
    /// while drivers rarely rely on the behavior of "Level-triggered" interrupt
    /// which keeps interrupting while interrupt pending signal is asserted.
    pub fn is_interrupting(&self) -> bool {
        self.interrupting
    }

    fn update_iir(&mut self) {
        let rx_ip = (self.ier & IER_RXINT_BIT) != 0 && self.rbr != 0;
        let thre_ip = (self.ier & IER_THREINT_BIT) != 0 && self.thr == 0;

        // Which should be prioritized RX interrupt or THRE interrupt?
        if rx_ip {
            self.iir = IIR_RD_AVAILABLE;
        } else if thre_ip {
            self.iir = IIR_THR_EMPTY;
        } else {
            self.iir = IIR_NO_INTERRUPT;
        }
    }

    /// Loads register content
    ///
    /// # Arguments
    /// * `address`
    pub fn load(&mut self, address: u64) -> u8 {
        //println!("UART Load AD:{:X}", address);
        match address {
            0x10000000 => match (self.lcr >> 7) == 0 {
                true => {
                    let rbr = self.rbr;
                    self.rbr = 0;
                    self.lsr &= !LSR_DATA_AVAILABLE;
                    self.update_iir();
                    rbr
                }
                false => 0, // @TODO: Implement properly
            },
            0x10000001 => match (self.lcr >> 7) == 0 {
                true => self.ier,
                false => 0, // @TODO: Implement properly
            },
            0x10000002 => self.iir,
            0x10000003 => self.lcr,
            0x10000004 => self.mcr,
            0x10000005 => self.lsr,
            0x10000007 => self.scr,
            _ => 0,
        }
    }

    /// Stores register content
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub fn store(&mut self, address: u64, value: u8) {
        //println!("UART Store AD:{:X} VAL:{:X}", address, value);
        match address {
            // Transfer Holding Register
            0x10000000 => match (self.lcr >> 7) == 0 {
                true => {
                    self.thr = value;
                    self.lsr &= !LSR_THR_EMPTY;
                    self.update_iir();
                }
                false => {} // @TODO: Implement properly
            },
            0x10000001 => match (self.lcr >> 7) == 0 {
                true => {
                    // This bahavior isn't written in the data sheet
                    // but some drivers seem to rely on it.
                    if (self.ier & IER_THREINT_BIT) == 0
                        && (value & IER_THREINT_BIT) != 0
                        && self.thr == 0
                    {
                        self.thre_ip = true;
                    }
                    self.ier = value;
                    self.update_iir();
                }
                false => {} // @TODO: Implement properly
            },
            0x10000003 => {
                self.lcr = value;
            }
            0x10000004 => {
                self.mcr = value;
            }
            0x10000007 => {
                self.scr = value;
            }
            _ => {}
        };
    }

    /// Returns mutable reference to `Terminal`.
    pub fn get_mut_terminal(&mut self) -> &mut Box<dyn Terminal> {
        &mut self.terminal
    }
}
