extern crate fnv;

#[cfg(feature = "std")]
use self::fnv::FnvHashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::btree_map::BTreeMap as FnvHashMap;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// ELF header
pub struct Header {
    pub e_width: u8, // 32 or 64
    _e_class: u8,
    _e_endian: u8,
    _e_elf_version: u8,
    _e_osabi: u8,
    _e_abi_version: u8,
    _e_type: u16,
    _e_machine: u16,
    _e_version: u32,
    pub e_entry: u64,
    _e_phoff: u64,
    e_shoff: u64,
    _e_flags: u32,
    _e_ehsize: u16,
    _e_phentsize: u16,
    _e_phnum: u16,
    _e_shentsize: u16,
    e_shnum: u16,
    _e_shstrndx: u16,
}

/// ELF program header
pub struct _ProgramHeader {
    _p_type: u32,
    _p_flags: u32,
    _p_offset: u64,
    _p_vaddr: u64,
    _p_paddr: u64,
    _p_filesz: u64,
    _p_memsz: u64,
    _p_align: u64,
}

/// ELF section header
#[derive(Debug)]
pub struct SectionHeader {
    #[allow(dead_code)]
    sh_name: u32,
    pub sh_type: u32,
    _sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    _sh_link: u32,
    _sh_info: u32,
    _sh_addralign: u64,
    _sh_entsize: u64,
}

/// ELF symbol table entry
pub struct SymbolEntry {
    st_name: u32,
    st_info: u8,
    _st_other: u8,
    _st_shndx: u16,
    st_value: u64,
    _st_size: u64,
}

/// ELF file analyzer
pub struct ElfAnalyzer {
    data: Vec<u8>,
}

impl ElfAnalyzer {
    /// Creates a new `ElfAnalyzer`.
    ///
    /// # Arguments
    /// * `data` ELF file content binary
    pub fn new(data: &[u8]) -> Self {
        ElfAnalyzer {
            data: data.to_vec(),
        }
    }

    /// Checks if ELF file content is valid
    // @TODO: Validate more precisely
    pub fn validate(&self) -> bool {
        // check ELF magic number
        if self.data.len() < 4
            || self.data[0] != 0x7f
            || self.data[1] != 0x45
            || self.data[2] != 0x4c
            || self.data[3] != 0x46
        {
            return false;
        }
        true
    }

    /// Reads ELF header
    pub fn read_header(&self) -> Header {
        let e_class = self.read_byte(4);

        let e_width = match e_class {
            1 => 32,
            2 => 64,
            _ => panic!("Unknown e_class:{e_class:X}"),
        };

        let e_endian = self.read_byte(5);
        let e_elf_version = self.read_byte(6);
        let e_osabi = self.read_byte(7);
        let e_abi_version = self.read_byte(8);

        let mut offset = 0x10;

        let e_type = self.read_halfword(offset);
        offset += 2;

        let e_machine = self.read_halfword(offset);
        offset += 2;

        let e_version = self.read_word(offset);
        offset += 4;

        let e_entry = match e_width {
            64 => {
                let data = self.read_doubleword(offset);
                offset += 8;
                data
            }
            _ => {
                let data = self.read_word(offset);
                offset += 4;
                data as u64
            }
        };

        let e_phoff = match e_width {
            64 => {
                let data = self.read_doubleword(offset);
                offset += 8;
                data
            }
            _ => {
                let data = self.read_word(offset);
                offset += 4;
                data as u64
            }
        };

        let e_shoff = match e_width {
            64 => {
                let data = self.read_doubleword(offset);
                offset += 8;
                data
            }
            _ => {
                let data = self.read_word(offset);
                offset += 4;
                data as u64
            }
        };

        let e_flags = self.read_word(offset);
        offset += 4;

        let e_ehsize = self.read_halfword(offset);
        offset += 2;

        let e_phentsize = self.read_halfword(offset);
        offset += 2;

        let e_phnum = self.read_halfword(offset);
        offset += 2;

        let e_shentsize = self.read_halfword(offset);
        offset += 2;

        let e_shnum = self.read_halfword(offset);
        offset += 2;

        let e_shstrndx = self.read_halfword(offset);
        //offset += 2;

        /*
        println!("ELF:{}", e_width);
        println!("e_endian:{:X}", e_endian);
        println!("e_elf_version:{:X}", e_elf_version);
        println!("e_osabi:{:X}", e_osabi);
        println!("e_abi_version:{:X}", e_abi_version);
        println!("e_type:{:X}", e_type);
        println!("e_machine:{:X}", e_machine);
        println!("e_version:{:X}", e_version);
        println!("e_entry:{:X}", e_entry);
        println!("e_phoff:{:X}", e_phoff);
        println!("e_shoff:{:X}", e_shoff);
        println!("e_flags:{:X}", e_flags);
        println!("e_ehsize:{:X}", e_ehsize);
        println!("e_phentsize:{:X}", e_phentsize);
        println!("e_phnum:{:X}", e_phnum);
        println!("e_shentsize:{:X}", e_shentsize);
        println!("e_shnum:{:X}", e_shnum);
        println!("e_shstrndx:{:X}", e_shstrndx);
        */

        Header {
            e_width,
            _e_class: e_class,
            _e_endian: e_endian,
            _e_elf_version: e_elf_version,
            _e_osabi: e_osabi,
            _e_abi_version: e_abi_version,
            _e_type: e_type,
            _e_machine: e_machine,
            _e_version: e_version,
            e_entry,
            _e_phoff: e_phoff,
            e_shoff,
            _e_flags: e_flags,
            _e_ehsize: e_ehsize,
            _e_phentsize: e_phentsize,
            _e_phnum: e_phnum,
            _e_shentsize: e_shentsize,
            e_shnum,
            _e_shstrndx: e_shstrndx,
        }
    }

    /// Reads ELF program headers
    ///
    /// # Arguments
    /// * `header`
    pub fn _read_program_headers(&self, header: &Header) -> Vec<_ProgramHeader> {
        let mut headers = Vec::new();
        let mut offset = header._e_phoff as usize;
        for _i in 0..header._e_phnum {
            let p_type = self.read_word(offset);
            offset += 4;

            let mut p_flags = 0;
            if header.e_width == 64 {
                p_flags = self.read_word(offset);
                offset += 4;
            }

            let p_offset = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let p_vaddr = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let p_paddr = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let p_filesz = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let p_memsz = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            if header.e_width == 32 {
                p_flags = self.read_word(offset);
                offset += 4;
            }

            let p_align = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            /*
            println!("");
            println!("Program:{:X}", i);
            println!("p_type:{:X}", p_type);
            println!("p_flags:{:X}", p_flags);
            println!("p_offset:{:X}", p_offset);
            println!("p_vaddr:{:X}", p_vaddr);
            println!("p_paddr:{:X}", p_paddr);
            println!("p_filesz:{:X}", p_filesz);
            println!("p_memsz:{:X}", p_memsz);
            println!("p_align:{:X}", p_align);
            println!("p_align:{:X}", p_align);
            */

            headers.push(_ProgramHeader {
                _p_type: p_type,
                _p_flags: p_flags,
                _p_offset: p_offset,
                _p_vaddr: p_vaddr,
                _p_paddr: p_paddr,
                _p_filesz: p_filesz,
                _p_memsz: p_memsz,
                _p_align: p_align,
            });
        }

        headers
    }

    /// Reads ELF section headers
    ///
    /// # Arguments
    /// * `header`
    pub fn read_section_headers(&self, header: &Header) -> Vec<SectionHeader> {
        let mut headers = Vec::new();
        let mut offset = header.e_shoff as usize;
        for _i in 0..header.e_shnum {
            let sh_name = self.read_word(offset);
            offset += 4;

            let sh_type = self.read_word(offset);
            offset += 4;

            let sh_flags = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_addr = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_offset = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_size = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_link = self.read_word(offset);
            offset += 4;

            let sh_info = self.read_word(offset);
            offset += 4;

            let sh_addralign = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_entsize = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            /*
            println!("");
            println!("Section:{:X}", _i);
            println!("sh_name:{:X}", sh_name);
            println!("sh_type:{:X}", sh_type);
            println!("sh_flags:{:X}", sh_flags);
            println!("sh_addr:{:X}", sh_addr);
            println!("sh_offset:{:X}", sh_offset);
            println!("sh_size:{:X}", sh_size);
            println!("sh_link:{:X}", sh_link);
            println!("sh_info:{:X}", sh_info);
            println!("sh_addralign:{:X}", sh_addralign);
            println!("sh_entsize:{:X}", sh_entsize);
            */

            headers.push(SectionHeader {
                sh_name,
                sh_type,
                _sh_flags: sh_flags,
                sh_addr,
                sh_offset,
                sh_size,
                _sh_link: sh_link,
                _sh_info: sh_info,
                _sh_addralign: sh_addralign,
                _sh_entsize: sh_entsize,
            });
        }

        headers
    }

    /// Reads symbol entries of symbol table sections
    ///
    /// # Arguments
    /// * `Terminal`
    /// * `symbol_table_section_headers`
    pub fn read_symbol_entries(
        &self,
        header: &Header,
        symbol_table_section_headers: &Vec<&SectionHeader>,
    ) -> Vec<SymbolEntry> {
        let mut entries = Vec::new();
        for section_header in symbol_table_section_headers {
            let sh_offset = section_header.sh_offset;
            let sh_size = section_header.sh_size;

            let mut offset = sh_offset as usize;

            let entry_size = match header.e_width {
                64 => 24,
                32 => 16,
                _ => panic!("Not happen"),
            };

            for _j in 0..(sh_size / entry_size) {
                let st_name;
                let st_info;
                let _st_other;
                let _st_shndx;
                let st_value;
                let _st_size;

                match header.e_width {
                    64 => {
                        st_name = self.read_word(offset);
                        offset += 4;

                        st_info = self.read_byte(offset);
                        offset += 1;

                        _st_other = self.read_byte(offset);
                        offset += 1;

                        _st_shndx = self.read_halfword(offset);
                        offset += 2;

                        st_value = self.read_doubleword(offset);
                        offset += 8;

                        _st_size = self.read_doubleword(offset);
                        offset += 8;
                    }
                    32 => {
                        st_name = self.read_word(offset);
                        offset += 4;

                        st_value = self.read_word(offset) as u64;
                        offset += 4;

                        _st_size = self.read_word(offset) as u64;
                        offset += 4;

                        st_info = self.read_byte(offset);
                        offset += 1;

                        _st_other = self.read_byte(offset);
                        offset += 1;

                        _st_shndx = self.read_halfword(offset);
                        offset += 2;
                    }
                    _ => panic!("No happen"),
                };

                /*
                println!("Symbol: {}", _j);
                println!("st_name: {:X}", st_name);
                println!("st_info: {:X}", st_info);
                println!("st_other: {:X}", _st_other);
                println!("st_shndx: {:X}", _st_shndx);
                println!("st_value: {:X}", st_value);
                println!("st_size: {:X}", _st_size);
                println!("");
                */

                entries.push(SymbolEntry {
                    st_name,
                    st_info,
                    _st_other,
                    _st_shndx,
                    st_value,
                    _st_size,
                });
            }
        }
        entries
    }

    /// Reads strings from a string table section
    ///
    /// # Arguments
    /// * `section_header` The header of the string table section
    /// * `index` Offset in the string table section
    fn read_strings(&self, section_header: &SectionHeader, index: u64) -> String {
        let sh_offset = section_header.sh_offset;
        let sh_size = section_header.sh_size;
        let mut pos = 0;
        let mut symbol = String::new();
        loop {
            let addr = sh_offset + index + pos;
            if addr >= sh_offset + sh_size {
                break;
            }
            let value = self.read_byte(addr as usize);
            if value == 0 {
                break;
            }
            symbol.push(value as char);
            pos += 1;
        }
        symbol
    }

    /// Creates a symbol - virtual address mapping from symbol entries
    /// and a string table section.
    ///
    /// # Arguments
    /// * `entries` Symbol entries
    /// * `string_table_section_header` The header of the string table section
    pub fn create_symbol_map(
        &self,
        entries: &Vec<SymbolEntry>,
        string_table_section_header: &SectionHeader,
    ) -> FnvHashMap<String, u64> {
        let mut map = FnvHashMap::default();
        for entry in entries {
            let st_info = entry.st_info;
            let st_name = entry.st_name;
            let st_value = entry.st_value;

            // Stores only function and notype symbol
            if (st_info & 0x2) != 0x2 && (st_info & 0xf) != 0 {
                continue;
            }

            let symbol = self.read_strings(string_table_section_header, st_name as u64);

            if !symbol.is_empty() {
                //println!("{} {:0x}", symbol, st_value);
                map.insert(symbol, st_value);
            }
        }
        map
    }

    /// Reads a byte from ELF file content
    ///
    /// # Arguments
    /// * `offset`
    pub fn read_byte(&self, offset: usize) -> u8 {
        self.data[offset]
    }

    /// Reads two bytes from ELF file content
    ///
    /// # Arguments
    /// * `offset`
    fn read_halfword(&self, offset: usize) -> u16 {
        let mut data = 0;
        for i in 0..2 {
            data |= (self.read_byte(offset + i) as u16) << (8 * i);
        }
        data
    }

    /// Reads four bytes from ELF file content
    ///
    /// # Arguments
    /// * `offset`
    fn read_word(&self, offset: usize) -> u32 {
        let mut data = 0;
        for i in 0..4 {
            data |= (self.read_byte(offset + i) as u32) << (8 * i);
        }
        data
    }

    /// Reads eight bytes from ELF file content
    ///
    /// # Arguments
    /// * `offset`
    fn read_doubleword(&self, offset: usize) -> u64 {
        let mut data = 0;
        for i in 0..8 {
            data |= (self.read_byte(offset + i) as u64) << (8 * i);
        }
        data
    }
}

/// Hand-assembled ELF64 fixtures for emulator tests. Field offsets follow the
/// System V gABI ELF64 layout, so these bytes are an oracle independent of the
/// analyzer under test.
#[cfg(test)]
pub(crate) mod test_elf {
    pub(crate) struct TestSymbol {
        pub name: &'static str,
        pub value: u64,
        /// st_info (binding << 4 | type), e.g. 0x10 = GLOBAL|NOTYPE, 0x12 = GLOBAL|FUNC
        pub info: u8,
        pub size: u64,
    }

    /// Builds a minimal but well-formed RV64 ELF: `.text` loaded at
    /// 0x8000_0000 with the given instruction words, plus a symbol table.
    pub(crate) fn build_elf64(text: &[u32], symbols: &[TestSymbol]) -> Vec<u8> {
        const TEXT_ADDR: u64 = 0x8000_0000;
        let text_bytes: Vec<u8> = text.iter().flat_map(|w| w.to_le_bytes()).collect();

        let align8 = |offset: usize| offset.div_ceil(8) * 8;

        let text_offset = 0x40; // right after the 64-byte ELF header
        let symtab_offset = align8(text_offset + text_bytes.len());
        let symtab_size = 24 * (symbols.len() + 1); // null entry + symbols

        // .strtab: leading NUL, then NUL-terminated names
        let mut strtab = vec![0u8];
        let mut name_offsets = Vec::new();
        for symbol in symbols {
            name_offsets.push(strtab.len() as u32);
            strtab.extend_from_slice(symbol.name.as_bytes());
            strtab.push(0);
        }
        let strtab_offset = symtab_offset + symtab_size;

        let shstrtab: &[u8] = b"\0.text\0.symtab\0.strtab\0.shstrtab\0";
        let shstrtab_offset = strtab_offset + strtab.len();
        let shoff = align8(shstrtab_offset + shstrtab.len());

        let mut elf = Vec::new();
        // ELF header
        elf.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        elf.extend_from_slice(&2u16.to_le_bytes()); // e_type = EXEC
        elf.extend_from_slice(&0xf3u16.to_le_bytes()); // e_machine = RISC-V
        elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
        elf.extend_from_slice(&TEXT_ADDR.to_le_bytes()); // e_entry
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
        elf.extend_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
        elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
        elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
        elf.extend_from_slice(&5u16.to_le_bytes()); // e_shnum
        elf.extend_from_slice(&4u16.to_le_bytes()); // e_shstrndx
        assert_eq!(elf.len(), 0x40);

        // .text content
        elf.extend_from_slice(&text_bytes);
        elf.resize(symtab_offset, 0);

        // .symtab: null entry then the given symbols (st_shndx = .text)
        elf.extend_from_slice(&[0u8; 24]);
        for (symbol, name_offset) in symbols.iter().zip(&name_offsets) {
            elf.extend_from_slice(&name_offset.to_le_bytes());
            elf.push(symbol.info);
            elf.push(0); // st_other
            elf.extend_from_slice(&1u16.to_le_bytes()); // st_shndx
            elf.extend_from_slice(&symbol.value.to_le_bytes());
            elf.extend_from_slice(&symbol.size.to_le_bytes());
        }

        elf.extend_from_slice(&strtab);
        elf.extend_from_slice(shstrtab);
        elf.resize(shoff, 0);

        let mut push_shdr = |name: u32,
                             sh_type: u32,
                             flags: u64,
                             addr: u64,
                             offset: u64,
                             size: u64,
                             link: u32,
                             info: u32,
                             addralign: u64,
                             entsize: u64| {
            let elf = &mut elf;
            elf.extend_from_slice(&name.to_le_bytes());
            elf.extend_from_slice(&sh_type.to_le_bytes());
            elf.extend_from_slice(&flags.to_le_bytes());
            elf.extend_from_slice(&addr.to_le_bytes());
            elf.extend_from_slice(&offset.to_le_bytes());
            elf.extend_from_slice(&size.to_le_bytes());
            elf.extend_from_slice(&link.to_le_bytes());
            elf.extend_from_slice(&info.to_le_bytes());
            elf.extend_from_slice(&addralign.to_le_bytes());
            elf.extend_from_slice(&entsize.to_le_bytes());
        };

        push_shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0); // SHT_NULL
        push_shdr(
            1,   // ".text"
            1,   // SHT_PROGBITS
            0x6, // ALLOC | EXECINSTR
            TEXT_ADDR,
            text_offset as u64,
            text_bytes.len() as u64,
            0,
            0,
            4,
            0,
        );
        push_shdr(
            7, // ".symtab"
            2, // SHT_SYMTAB
            0,
            0,
            symtab_offset as u64,
            symtab_size as u64,
            3, // link to .strtab
            1, // one local symbol (the null entry)
            8,
            24,
        );
        push_shdr(
            15, // ".strtab"
            3,  // SHT_STRTAB
            0,
            0,
            strtab_offset as u64,
            strtab.len() as u64,
            0,
            0,
            1,
            0,
        );
        push_shdr(
            23, // ".shstrtab"
            3,
            0,
            0,
            shstrtab_offset as u64,
            shstrtab.len() as u64,
            0,
            0,
            1,
            0,
        );

        elf
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test-only assertions")]
mod tests {
    use super::test_elf::{build_elf64, TestSymbol};
    use super::*;

    fn fixture() -> ElfAnalyzer {
        let elf = build_elf64(
            &[0x0010_0093, 0x0000_006f], // addi x1,x0,1 ; j .
            &[
                TestSymbol {
                    name: "_start",
                    value: 0x8000_0000,
                    info: 0x12, // GLOBAL | FUNC
                    size: 8,
                },
                TestSymbol {
                    name: "tohost",
                    value: 0x8000_3000,
                    info: 0x10, // GLOBAL | NOTYPE
                    size: 8,
                },
                TestSymbol {
                    name: "an_object",
                    value: 0x8000_4000,
                    info: 0x11, // GLOBAL | OBJECT: filtered from the symbol map
                    size: 8,
                },
            ],
        );
        ElfAnalyzer::new(&elf)
    }

    #[test]
    fn validate_requires_the_elf_magic() {
        assert!(fixture().validate());
        assert!(!ElfAnalyzer::new(&[]).validate());
        assert!(!ElfAnalyzer::new(&[0x7f, b'E', b'L']).validate());
        assert!(!ElfAnalyzer::new(b"\x7fBAD").validate());
    }

    #[test]
    fn read_header_extracts_the_elf64_fields() {
        let header = fixture().read_header();
        assert_eq!(header.e_width, 64);
        assert_eq!(header.e_entry, 0x8000_0000);
        assert_eq!(header.e_shnum, 5);
    }

    #[test]
    #[should_panic(expected = "Unknown e_class")]
    fn read_header_rejects_unknown_classes() {
        let mut elf = build_elf64(&[], &[]);
        elf[4] = 9; // neither ELFCLASS32 nor ELFCLASS64
        let _ = ElfAnalyzer::new(&elf).read_header();
    }

    #[test]
    fn section_headers_describe_text_symtab_and_strtab() {
        let analyzer = fixture();
        let header = analyzer.read_header();
        let sections = analyzer.read_section_headers(&header);
        assert_eq!(sections.len(), 5);

        assert_eq!(sections[0].sh_type, 0);
        // .text: PROGBITS at the load address with 2 instruction words
        assert_eq!(sections[1].sh_type, 1);
        assert_eq!(sections[1].sh_addr, 0x8000_0000);
        assert_eq!(sections[1].sh_offset, 0x40);
        assert_eq!(sections[1].sh_size, 8);
        // .symtab: 4 entries of 24 bytes (null + 3 symbols)
        assert_eq!(sections[2].sh_type, 2);
        assert_eq!(sections[2].sh_size, 96);
        // both string tables
        assert_eq!(sections[3].sh_type, 3);
        assert_eq!(sections[4].sh_type, 3);
    }

    #[test]
    fn symbol_map_keeps_functions_and_notype_but_drops_objects() {
        let analyzer = fixture();
        let header = analyzer.read_header();
        let sections = analyzer.read_section_headers(&header);
        let symtabs: Vec<&SectionHeader> = sections.iter().filter(|s| s.sh_type == 2).collect();
        let strtabs: Vec<&SectionHeader> = sections.iter().filter(|s| s.sh_type == 3).collect();

        let entries = analyzer.read_symbol_entries(&header, &symtabs);
        assert_eq!(entries.len(), 4, "null entry plus three symbols");

        let map = analyzer.create_symbol_map(&entries, strtabs[0]);
        assert_eq!(map.get("_start"), Some(&0x8000_0000));
        assert_eq!(map.get("tohost"), Some(&0x8000_3000));
        assert_eq!(
            map.get("an_object"),
            None,
            "STT_OBJECT symbols are excluded from the map"
        );
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn read_strings_stops_at_nul_and_section_end() {
        let analyzer = fixture();
        let header = analyzer.read_header();
        let sections = analyzer.read_section_headers(&header);
        let strtab = sections
            .iter()
            .find(|s| s.sh_type == 3)
            .expect("strtab present");

        // Offset 1 is "_start" (the table begins with a NUL byte)
        assert_eq!(analyzer.read_strings(strtab, 1), "_start");
        // Reading from the final NUL yields an empty string
        assert_eq!(analyzer.read_strings(strtab, strtab.sh_size - 1), "");
        // Reading at the section end yields an empty string instead of overrunning
        assert_eq!(analyzer.read_strings(strtab, strtab.sh_size), "");
    }
}
